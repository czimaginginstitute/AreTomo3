#--------------------------------------------------------------
# This makefile needs to be run in its containing directory.
# The following setup ensures we are indeed in Util.
#--------------------------------------------------------------
PARENT_DIR = $(shell pwd)/..
UTIL_DIR = $(PARENT_DIR)/Util
INC_DIR = $(PARENT_DIR)/Include
LIB_DIR = $(PARENT_DIR)/Lib
#-------------------
SRCS = $(wildcard *.cpp)
HDRS = $(patsubst %.cpp, %.h, $(SRCS))
OBJS = $(patsubst %.cpp, %.o, $(SRCS))
#-------------------
OUT = libutil.a

CC = g++ -std=c++11
CFLAG = -c -g -pthread -m64

all: $(OBJS)
	@echo create library $(OUT) and move to $(LIB_DIR)
	@echo copy headers to $(INC_DIR)/Util	
	@ar rcs $(OUT) $(OBJS)
	@mv $(OUT) $(LIB_DIR)
	@cp $(HDRS) $(INC_DIR)/Util

%.o: %.cpp
	$(CC) $(CFLAG) $< -o $@

list:
	@echo $(OBJS)

clean:
	@echo delete all object files and temporary files	
	@rm -f $(OBJS)
	@rm -f $(UTIL_DIR)/*.*~
	@rm -f $(UTIL_DIR)/*~
