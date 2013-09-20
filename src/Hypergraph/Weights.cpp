#include "Weights.h"
DEFINE_string(weight_file, "", "svector weight file with translation params"); // was 3
//static const bool weight_dummy = RegisterFlagValidator(&FLAGS_weight_file, &ValidateFile);


wvector * load_weights_from_file(const char * file)  {
  fstream input(file, ios::in );

  char buf[1000];
  input.getline(buf, 100000);
  string s(buf);
  return svector_from_str<int, double>(s);
}

wvector * cmd_weights()  {
  return load_weights_from_file(FLAGS_weight_file.c_str());
}
