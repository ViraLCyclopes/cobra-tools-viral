textures = {
	# 'Metallic_Roughness_Opaque_Emissive': {
	# 	# assuming this is more common
	# 	# "pRoughnessPackedTexture": {"R": "MT", "G": "SP", "B": "RN", "A": "FO"},
	# },
	# 'Metallic_Roughness_Opaque_BC7': {
	# 	'pBaseColourTexture': {"RGB": "BC", "A": "MT"},
	# 	# pNormalTexture is actually {"RG": "NM", "B": "SP", "A": "RN"}
	# 	# TODO: Handle RGB vs RG/B split of pNormalTexture
	# 	'pNormalTexture': {"RGB": "NM", "A": "RN"}
	# },
	# VERIFIED against shipped JWE3 assets 2026-09-13. Without this override the
	# default texchannels map reads R as MT (metallic), nothing drives alpha,
	# and every leaf card imports as an opaque rectangle.
	#
	# Evidence, 4/4 Foliage_Clip materials sampled from Tree_Ceiba, Tree_Baobab,
	# Tree_Paleo_Gingko and Tree_Oak:
	#   pBaseColourTexture      A is CONSTANT 255 (distinct=1) - not the mask
	#   pRoughnessPackedTexture R is bimodal (74% black, 11% white) and reads as
	#                           leaf silhouettes when viewed
	# Same packing as JWE2, which is where these values come from.
	'Foliage_Clip': {
		"pRoughnessPackedTexture": {"R": "OP", "G": "RN", "B": "SP", "A": "TR"},
	},
	# 'Foliage_ClipNoDisplacement': {
	# 	"pRoughnessPackedTexture": {"R": "OP", "G": "RN", "B": "SP", "A": "TR"},
	# },
	# 'Foliage_ClipTexcoordWeight': {
	# 	"pRoughnessPackedTexture": {"R": "OP", "G": "RN", "B": "SP", "A": "TR"},
	# },
	# 'Foliage_Opaque': {
	# },
	# 'Foliage_Billboard': {
	# 	# "pNormalTexture" # not sure if A is actually AO, might be unused (1.0)
	# },
	# 'Glass_TexturedNormalsTwoSided': {
	# 	'pNormalTexture': {"RG": "NM"}
	# }
}
