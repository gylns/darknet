#ifndef PARSER_H
#define PARSER_H
#include "network.h"

network parse_network_cfg(char *filename);
network parse_network_cfg_custom(char *filename, int batch);
list *read_cfg_file(FILE *file);
list *read_cfg_mem(char *buffer);
network parse_network_cfg_custom_file(FILE *fp, int batch);
network parse_network_cfg_custom_mem(char *buffer, int batch);
void save_network(network net, char *filename);
void save_weights(network net, char *filename);
void save_weights_cfgbuf(network net, char *filename, const char *cfgbuf, size_t cfgsize);
void save_weights_upto(network net, char *filename, int cutoff);
void save_weights_cfgbuf_upto(network net, char *filename, int cutoff, const char *cfgbuf, size_t cfgsize);
void save_weights_double(network net, char *filename);
void load_weights(network *net, char *filename);
void load_weights_mem(network *net, char *buffer);
void load_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto_file(network *net, FILE *fp, int cutoff);

#endif
