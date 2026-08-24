# START_GLOBALS
import logging
import struct

from generated.base_struct import BaseStruct

# ACL's compressed_database serialization tag
ACL_DATABASE_TAG = 0xAC11DB01
# raw_buffer_header is uint32 size + uint32 hash, then the tag
ACL_DATABASE_MIN_SIZE = 32

# END_GLOBALS


class KeysReader(BaseStruct):

	# START_CLASS

	@classmethod
	def read_fields(cls, stream, instance):
		"""Read the ACL compressed_database that sits between the ManiInfo array and
		the name buffer, if this bundle has one.

		JWE3 clips are ACL database-stripped: the retained frames live in the clip and
		the per-frame detail in database bulk blocks shipped as paired LOD streams. The
		database's own header sits here, at the tail of buffer 0, immediately after
		`mani_count * sizeof(ManiInfo)` bytes.

		This used to be modelled as a `CompressedHeader` struct of pointers, gated on
		`dtype.has_list == 3`. That was a guess which only ever worked because it
		happened to consume the same number of bytes, and it made the database
		mandatory: a bundle with the database removed failed to parse at all, because
		the reader consumed the name buffer in its place and then ran off the end of
		the file. It also mis-fired on the dtype 48/49 bundle, which has has_list == 1
		yet still carries a database.

		Detect the blob by ACL's tag and take it verbatim instead. Being byte-preserving
		matters as much here as it does in MANI.py: re-serialising a JWE3 manis through
		a guessed schema is what silently dropped this blob before.
		"""
		instance.io_start = stream.tell()
		instance.data = None
		head = stream.read(12)
		stream.seek(instance.io_start)
		if len(head) == 12:
			size, _hash, tag = struct.unpack("<III", head)
			if tag == ACL_DATABASE_TAG and size >= ACL_DATABASE_MIN_SIZE:
				# the blob is followed by padding up to 16, which belongs to buffer 0.
				# Reading only `size` leaves the stream 8 bytes early on any database
				# whose size is not a multiple of 16 (Indoraptor f0ad5573 is 296), and
				# the name/hash buffer is then parsed out of that padding.
				padded = size + (-size % 16)
				instance.data = stream.read(padded)
				if len(instance.data) != padded:
					raise BufferError(
						f"ACL compressed_database claims {size} bytes ({padded} padded) "
						f"but only {len(instance.data)} were available")
		logging.debug(f"CompressedHeaderReader read {0 if instance.data is None else len(instance.data)} bytes")
		instance.io_size = stream.tell() - instance.io_start

	@classmethod
	def write_fields(cls, stream, instance):
		"""Write the verbatim database blob back, or nothing if there is none.

		`read_fields` sets `data` to a bytes blob, or None when the bundle carries no ACL
		database. A *freshly constructed* instance has neither: the schema declares
		`data` as an abstract `CompressedHeader`, so `set_defaults` builds a struct there,
		and writing it raised `a bytes-like object is required, not 'CompressedHeader'` on
		every Blender export.

		Writing nothing in that case is what the format actually does - the shipped
		`hatcheryexitcamera` bundle is dtype 0 with no database and occupies zero bytes
		in this region, reading back as `data is None`.
		"""
		instance.io_start = stream.tell()
		if isinstance(instance.data, (bytes, bytearray, memoryview)):
			stream.write(instance.data)
		elif instance.data is not None:
			logging.debug(
				f"CompressedHeaderReader has no database blob "
				f"({type(instance.data).__name__}), writing nothing")
		instance.io_size = stream.tell() - instance.io_start
