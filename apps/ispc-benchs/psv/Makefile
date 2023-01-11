CXX=parsimony
CXXFLAGS=-Wall -O3 -march=native -mprefer-vector-width=512 --Xtmp tmp

PATHS= .
SRCS = $(foreach sdir,$(PATHS),$(wildcard $(sdir)/*.cpp))
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

LIB=libpsv.a

.PHONY: all clean
all: $(LIB)

$(LIB): $(OBJS)
	ar rc $(LIB) $(OBJS)
	ranlib $(LIB)

clean:
	rm -rf $(OBJS) $(LIB) tmp

%.o: %.cpp Math.h
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@
