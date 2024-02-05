PARENT_DIR = $(shell pwd)/..
PRJ_DIR = $(PARENT_DIR)/Mrcfile
INC_DIR	= $(PARENT_DIR)/Include
LIB_DIR  = $(PARENT_DIR)/Lib

SRCS    = Header/CLoadExtHeader.cpp \
	Header/CLoadMainHeader.cpp \
	Header/CMainHeader.cpp \
	Header/CSaveExtHeader.cpp \
	Header/CSaveMainHeader.cpp \
	Image/CLoadImage.cpp \
	Image/CSaveImage.cpp \
	Image/C4BitImage.cpp \
	App/CLoadMrc.cpp \
	App/CSaveMrc.cpp \
	App/CReviseExtHeader.cpp \
	App/CMrcScale.cpp \
	App/CVerticalFlip.cpp

OBJS	= $(patsubst %.cpp, %.o, $(SRCS))
OUT	= libmrcfile.a

CC = g++ -std=c++11
CFLAGS	= -m64 -c -g -pthread -IInclude -I$(INC_DIR)

all: $(OBJS)
	@ar rcs $(OUT) $(OBJS)
	@mv -f $(OUT) $(LIB_DIR)
	@cp -f Include/*.h $(INC_DIR)/Mrcfile
	@echo create library $(OUT) and move to $(LIB_DIR)

compile: $(OBJS)
	@echo Compilation has been done.

%.o: %.cpp
	@echo Compile $<
	@$(CC) $(CFLAGS) $< -o $@

clean:
	@rm -f $(OUT) $(OBJS) *.*~ makefile~
