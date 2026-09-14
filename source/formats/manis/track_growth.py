"""Append a custom ori/pos track to one JWE3 ACL clip.

Derived from the September 7 grown-track control (stock-rig spawn passed).
No schemas or readers change. A database rebuild preserves limb payloads.
Visible custom-bone motion still requires game verification.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

from generated.formats.manis import ManisFile
from generated.formats.ms2 import Ms2File
from generated.formats.manis.acl import decode_file, _write_jacl
from source.formats.manis.append import preamble_layout, block_extents
from source.formats.manis.channel_growth import channel_layout, rebuild
from source.formats.manis.database import locate_bulk, find_database, check_name_buffer
from source.formats.manis.splice import list_clip_blobs, read_blob_header
from source.formats.manis.limbs import block_layout, buffer_residue, rebuild_block
from source.formats.manis.bindpose import clip_defaults, write_jbind, NO_PARENT
from source.formats.manis.bonemask import find_mask, set_bits, read_mask, bones_in


COBRA = Path(__file__).resolve().parents[3]

def pad(data, alignment):
    return data + bytes(-len(data) % alignment)


def load(path):
    m = ManisFile(); m.load(str(path))
    assert (m.version, m.mani_version) == (262, 282)
    return m


def hash_name(name):
    h = 5381
    for c in name.lower().encode('utf8'):
        h = (h * 33 + c) & 0xffffffff
    return h


def grow_metadata(data, m, ci, bone, add=True, write_names=True):
    """Rebuild the counted channel/name arrays; copy all other bytes verbatim.

    The unchanged path reconstructs both arrays and must be byte-identical.
    Unknown count_related/repeat fields stay verbatim: shipped clips extend
    to tracks 178/179 while retaining 170/172 in those fields.
    """
    info = m.mani_infos[ci]
    track = int(info.target_bone_count)
    assert 0 < track < 254 and not info.dtype.use_ushort
    if add:
        existing_tracks = {int(t) for other in m.mani_infos for group in ('ori','pos','scl')
                           for n,t in zip(getattr(other.keys,group+'_bones_names'),
                                          getattr(other.keys,group+'_channel_to_bone')) if str(n)==bone}
        if existing_tracks and existing_tracks != {track}:
            raise ValueError(f'{bone}: clips require incompatible new track indices; export blocked')
    names = list(map(str, m.name_buffer.target_names))
    name_index = names.index(bone) if bone in names else len(names)
    plan = channel_layout(info)
    start, end = block_extents(m, locate_bulk(data)['low_offset'])[ci]
    block = data[start:end]
    pieces = {k: block[v[0]:v[0]+v[1]] for k, v in plan.items() if not k.startswith('_')}
    raw_info = bytearray(data[info.io_start:info.io_start+304])
    if add:
        for group, countoff, maxoff in (('pos',24,281),('ori',26,283)):
            count = plan['_counts'][group]
            low = int(getattr(info, group+'_bone_min'))
            high = int(getattr(info, group+'_bone_max'))
            assert low == 0 and high < track < 255 and raw_info[maxoff] == high
            assert struct.unpack_from('<H',raw_info,countoff)[0] == count
            pieces[group+'_names'] += struct.pack('<I',name_index)
            pieces[group+'_c2b'] += bytes([track])
            pieces[group+'_b2c'] += bytes([255])*(track-high-1)+bytes([count])
            struct.pack_into('<H',raw_info,countoff,count+1)
            raw_info[maxoff] = track
        assert struct.unpack_from('<Q',raw_info,42)[0] == track
        struct.pack_into('<Q',raw_info,42,track+1)
    order = [g+'_names' for g in ('pos','ori','scl','float')]
    order += [g+'_c2b' for g in ('pos','ori','scl')]
    order += [g+'_b2c' for g in ('pos','ori','scl')]
    region = pad(b''.join(pieces[k] for k in order),4)
    old_blob = plan['_padded']+128
    old_blob += -old_blob % 16
    assert block.find(bytes.fromhex('11ac11ac'),plan['_padded'])-8 == old_blob
    prefix = bytearray(block[plan['_padded']:old_blob])
    mask = find_mask(data,start+old_blob,track)
    assert mask is not None and start+plan['_padded'] <= mask < start+old_blob
    if add:
        relative_mask = mask-start-plan['_padded']
        struct.pack_into('<I',prefix,relative_mask+0x40,track+1)
        set_bits(prefix,relative_mask,track+1,[track])
    new_blob = len(region)+128
    new_blob += -new_blob % 16
    structure = bytes(prefix).rstrip(b'\0')
    assert len(structure) <= new_blob-len(region)
    replacement = region+structure+bytes(new_blob-len(region)-len(structure))+block[old_blob:]
    assert len(replacement)%16 == len(block)%16 == 0
    out = bytearray(data)
    out[start:end] = replacement
    out[info.io_start:info.io_start+304] = raw_info
    if not write_names:
        return bytes(out)

    # Buffer1 hashes followed by ZStrings, with name-string-relative align4.
    name_start = m.name_buffer.io_start
    name_end = name_start+m.name_buffer.io_size
    hashes = bytes(data[name_start:name_start+4*len(names)])
    strings = b''.join(n.encode('utf8')+b'\0' for n in names)
    assert hashes+pad(strings,4) == data[name_start:name_end]
    assert all(hash_name(n)==int(h) for n,h in zip(names,m.name_buffer.target_hashes))
    if add and bone not in names:
        hashes += struct.pack('<I',hash_name(bone))
        strings += bone.encode('utf8')+b'\0'
        root = preamble_layout(m)['root_start']
        assert struct.unpack_from('<H',out,root+2)[0] == 4*len(names)
        struct.pack_into('<H',out,root+2,4*(len(names)+1))
    out[name_start:name_end] = hashes+pad(strings,4)
    return bytes(out)


def clip_bind(info, ms2, count, add, bone):
    """Resolve each clip's track names independently, including foreign tracks."""
    candidates = [b for b in ms2.models_reader.bone_infos if any(str(x.name) == bone for x in b.bones)]
    if len(candidates) != 1:
        raise ValueError(f'Supplied MS2 must contain exactly one rig with {bone}; export your edited rig first')
    bi = candidates[0]
    rig = {str(b.name):(j,b) for j,b in enumerate(bi.bones)}
    mapping = {}
    for g in ('ori','pos','scl'):
        for name,track in zip(getattr(info.keys,g+'_bones_names'),getattr(info.keys,g+'_channel_to_bone')):
            name,track = str(name),int(track)
            assert track not in mapping or mapping[track] == name
            mapping[track] = name
    if add:
        mapping[count-1] = bone
    reverse = {n:t for t,n in mapping.items()}
    parents = np.full(count,NO_PARENT,dtype='<u4')
    values = np.zeros((count,10),dtype='<f4'); values[:,3]=1;values[:,7:]=1
    for t,n in mapping.items():
        if n not in rig:
            continue
        j,b = rig[n]
        values[t] = [b.rot.x,b.rot.y,b.rot.z,b.rot.w,b.loc.x,b.loc.y,b.loc.z,b.scale,b.scale,b.scale]
        parent = int(bi.parents[j])
        # Rig Edit bridges are absent from the animation skeleton. Follow the
        # hierarchy to the nearest named ancestor for compression error metrics.
        while parent != 65535:
            pname = str(bi.bones[parent].name)
            if pname in reverse:
                parents[t] = reverse[pname]
                break
            parent = int(bi.parents[parent])
    return parents,values


def encode(source, streams, headers, m, ms2, ci, add, work, bone):
    manifest=[]; indices=[]
    clip_index=0
    for si,st in enumerate(streams):
        if st.track_type != 12:
            continue
        extra = add and (clip_index in ci if isinstance(ci,set) else clip_index==ci)
        par,val = clip_bind(m.mani_infos[clip_index],ms2,st.values.shape[1],extra,bone)
        jp=work/f'clip.{si}.jacl';bp=work/f'clip.{si}.jbind'
        _write_jacl(str(jp),st.values,st.track_type,st.sample_rate)
        write_jbind(str(bp),par,clip_defaults(st.values,val))
        manifest.append(f'{jp}|{int(headers[si]["wrap_optimized"])}|{bp}')
        indices.append(si);clip_index+=1
    manifest_path=work/'manifest.txt';manifest_path.write_text('\n'.join(manifest),encoding='utf8')
    output=work/'out';output.mkdir()
    run=subprocess.run([str(COBRA/'bin/jwe3_acl_database.exe'),str(manifest_path),str(output),'--bind',str(work/f'clip.{indices[0]}.jbind')],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+'\n'+run.stderr
    print(run.stdout.strip(),flush=True)
    blobs=list_clip_blobs(source)
    new=[source[o:o+s] for o,s in blobs] # scalar blobs remain byte-identical
    for position,si in enumerate(indices):
        new[si]=(output/f'clip.{position}.blob').read_bytes()
        assert read_blob_header(new[si])['has_database']
    keys=source[:locate_bulk(source)['low_offset']]
    blocks=block_layout(keys,blobs,headers);base=buffer_residue(blobs)
    cursor=len(new)
    for b in reversed(blocks):
        cursor-=len(b['blobs'])
        keys=rebuild_block(keys,b,base,new[cursor:cursor+len(b['blobs'])])
    off,size=find_database(keys)
    keys=keys[:off]+pad((output/'database.bin').read_bytes(),16)+keys[off+size+(-size%16):]
    return keys+pad((output/'bulk_low.bin').read_bytes(),16)+pad((output/'bulk_medium.bin').read_bytes(),16)



def add_bone_track(source, output, clip, bone, ms2_path):
    """Create constant bind ori/pos channels; caller then splices authored keys."""
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError('Track growth requires a separate output file')
    data = Path(source).read_bytes()
    m = load(source)
    ci = [str(i.name) for i in m.mani_infos].index(clip)
    info = m.mani_infos[ci]
    if any(bone in list(map(str, getattr(info.keys, g+'_bones_names'))) for g in ('ori','pos','scl')):
        raise ValueError(f'{bone} already has a track in {clip}; use channel growth')
    assert grow_metadata(data, m, ci, bone, False) == data, 'track metadata null gate failed'
    streams = decode_file(str(source))
    originals = [s.values.copy() for s in streams]
    blobs = list_clip_blobs(data)
    headers = [read_blob_header(data, o) for o, _ in blobs]
    tf = [i for i,s in enumerate(streams) if s.track_type == 12]
    assert len(tf) == len(m.mani_infos)
    si = tf[ci]; track = int(info.target_bone_count)
    assert streams[si].values.shape[1] == track
    ms2 = Ms2File(); ms2.load(str(ms2_path), read_editable=False)
    _, bind = clip_bind(info, ms2, track+1, True, bone)
    if not np.allclose(bind[-1,7:], 1):
        raise ValueError('New-track exporter currently requires unit bind scale')
    extra = np.full((streams[si].values.shape[0],1,10),np.nan,dtype='<f4')
    extra[:,:,:7] = bind[None,-1:,:7]
    streams[si].values = np.concatenate((streams[si].values,extra),axis=1)
    metadata = grow_metadata(data,m,ci,bone)
    with tempfile.TemporaryDirectory(prefix='cobra_new_track_') as td:
        result = encode(metadata,streams,headers,m,ms2,ci,True,Path(td),bone)
    Path(output).write_bytes(result)
    got = load(output)
    assert check_name_buffer(str(output))[0]
    decoded = decode_file(str(output))
    assert int(got.mani_infos[ci].target_bone_count) == track+1
    assert np.allclose(decoded[si].values[:,-1,:7],bind[-1,:7],atol=1e-5)
    for old,new in zip(originals,decoded):
        values = new.values[:,:old.shape[1],:]
        assert values.shape == old.shape
        assert np.array_equal(np.isnan(values),np.isnan(old)), 'stripped set drift'
        if new.track_type != 12:
            assert np.array_equal(values,old,equal_nan=True)
            continue
        for lo,hi in ((4,7),(7,10)):
            valid=np.isfinite(old[:,:,lo:hi])
            assert np.max(np.abs(values[:,:,lo:hi][valid]-old[:,:,lo:hi][valid]),initial=0)<.01
        valid=np.isfinite(old[:,:,:4]).all(axis=-1)
        a=old[:,:,:4][valid].astype('f8');b=values[:,:,:4][valid].astype('f8')
        a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
        angles=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=-1)),0,1)))
        assert np.max(angles,initial=0)<.5
    return track
