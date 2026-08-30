from inspect import isclass

from generated.formats.motiongraph.structs.DataStreamResourceData import DataStreamResourceData
from generated.formats.motiongraph.structs.MotiongraphHeader import MotiongraphHeader
import generated.formats.ovl.versions as ovl_versions
from modules.formats.BaseFormat import MemStructLoader


class MotiongraphLoader(MemStructLoader):
	target_class = MotiongraphHeader
	extension = ".motiongraph"

	@property
	def motiongraph_rename_sound(self):
		return self.ovl.cfg.get("motiongraph_rename_sound", False)

	def create(self, file_path):
		raise NotImplementedError(f"Can't create {self.name}")

	def collect(self):
		self.context.recursion = {}
		if self.ovl.version >= 19:
			# structs are too different, doesn't register anim names, would break rename contents
			if ovl_versions.is_jwe(self.ovl):
				return
			super().collect()

	def get_audio_strings(self):
		def cond(x):
			try:
				return x[1].__name__ == "DataStreamResourceData"
			except:
				return False
		# condition_function = lambda x: hasattr(x[1], "__name__") and x[1].__name__ == "DataStreamResourceData"
		# condition_function = lambda x:  issubclass(x[1], DataStreamResourceData)
		for data_stream_resource_data in self.header.get_condition_fields(cond):
			if data_stream_resource_data.type.data in ("AudioEvent", "AudioLoopingEvent", "AudioBlend", "AudioRTPC"):
				yield data_stream_resource_data.ds_name.data

	def accept_string(self, in_str):
		"""Return True if string should receive replacement"""
		# Animation references carry a species separator. JWE2 uses @, eg.
		# Acrocanthosaurus@JumpAttackDefendFlankLeft; JWE3 uses $, eg.
		# Deinosuchus$Partial_Unconscious. Measured over two JWE3 motiongraphs,
		# all 1,319 animation references (<mani>, <anim_name>, <activity_name>)
		# contain $ and none contain @, so accepting only @ silently skipped
		# every one of them and rename contents did nothing on JWE3.
		# Neither separator appears in the other game's strings, so accepting
		# both is safe for JWE1/JWE2/PZ.
		if "@" in in_str or "$" in in_str:
			return True
		# sound events have neither, e.g. Acrocanthosaurus_FightReact
		return self.motiongraph_rename_sound
