# Generated by the protocol buffer compiler.  DO NOT EDIT!

from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)


import hypergraph_pb2

DESCRIPTOR = descriptor.FileDescriptor(
  name='translation.proto',
  package='',
  serialized_pb='\n\x11translation.proto\x1a\x10hypergraph.proto:%\n\x10\x66oreign_sentence\x12\x0b.Hypergraph\x18\x01 \x01(\t:(\n\x13reference_sentences\x12\x0b.Hypergraph\x18\x02 \x03(\t')


FOREIGN_SENTENCE_FIELD_NUMBER = 1
foreign_sentence = descriptor.FieldDescriptor(
  name='foreign_sentence', full_name='foreign_sentence', index=0,
  number=1, type=9, cpp_type=9, label=1,
  has_default_value=False, default_value=unicode("", "utf-8"),
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  options=None)
REFERENCE_SENTENCES_FIELD_NUMBER = 2
reference_sentences = descriptor.FieldDescriptor(
  name='reference_sentences', full_name='reference_sentences', index=1,
  number=2, type=9, cpp_type=9, label=3,
  has_default_value=False, default_value=[],
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  options=None)


hypergraph_pb2.Hypergraph.RegisterExtension(foreign_sentence)
hypergraph_pb2.Hypergraph.RegisterExtension(reference_sentences)
# @@protoc_insertion_point(module_scope)