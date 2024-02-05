PRJHOME = $(HOME)/CuProjs/TomoAlign
CUDAHOME = /usr/local/cuda
CUDAINC = $(CUDAHOME)/include
CUDALIB = $(CUDAHOME)/lib64
PRJINC = $(HOME)/Projs/Include
PRJLIB = $(HOME)/Projs/Lib
CUPRJINC = $(HOME)/CuProjs/Include
CUPRJLIB = $(HOME)/CuProjs/Lib
#-----------------------------
CUSRCS = ./Align/GCC1D.cu \
	 ./Align/GCoherence.cu \
	 ./Align/GIntegrateX.cu \
	 ./Align/GRemoveMean.cu \
	 ./Align/GSumLines.cu \
	 ./Align/GAddTwoLines.cu \
	 ../Util/GRotate2D.cu
CUCPPS = $(patsubst %.cu, %.cpp, $(CUSRCS))
#------------------------------------------
SRCS = ../Util/CSimpleFuncs.cpp \
	../Util/CParseArgs.cpp \
	../Util/CSaveTempMrc.cpp \
	../MrcUtil/CTomoStack.cpp \
	../MrcUtil/CLoadStack.cpp \
	../MrcUtil/CSaveStack.cpp \
	./Align/CAlignMain.cpp \
	./Align/CAlignParam.cpp \
	./Align/CFindTiltAxis.cpp \
	./Align/CGenLines.cpp \
	./Align/CLineBuf.cpp \
	./Align/CCorrTiltAxis.cpp \
	./CInput.cpp \
	./CClMain.cpp \
	$(CUCPPS)
OBJS = $(patsubst %.cpp, %.o, $(SRCS))
#-------------------------------------
CC = g++
CFLAG = -c -g -pthread -m64
NVCC = nvcc
CUFLAG = -O3 -arch=sm_20  -Xptxas -dlcm=ca 
#-----------------------------------------
cuda: $(CUCPPS)

compile: $(OBJS)

exe: $(OBJS)
	@g++ -g -pthread -m64 $(OBJS) \
	$(PRJLIB)/libmrcfile.a \
	$(PRJLIB)/libutil.a \
	$(CUPRJLIB)/libcuutil.a \
	$(CUPRJLIB)/libcuutilfft.a \
	-L$(CUDALIB) -L/usr/lib64 \
	-lcufft -lcudart -lcuda -ltiff -lc -lm -lpthread \
	-o CommonlineExe
	@echo CommonlineExe has been generated.

%.cpp: %.cu
	@$(NVCC) -cuda $(CUFLAG) -I$(PRJINC) -I$(CUPRJINC) $< -o $@
	@echo $< has been compiled.

%.o: %.cpp
	@$(CC) $(CFLAG) -I$(PRJINC) -I$(CUPRJINC) -I$(CUDAINC) \
		$< -o $@
	@echo $< has been compiled.

clean:
	@rm -f $(OBJS) $(CUCPPS) *.h~ makefile~ DfCorrOffline
	
