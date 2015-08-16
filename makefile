CC = g++-4.8	
PROG = main

$(PROG).x : $(PROG).cpp
	$(CC) $(PROG).cpp -std=gnu++11 -I/usr/local/lib -lm -lstdc++  -L/home/intronati/GSL/lib -lgsl -lgslcblas -o $(PROG).x

$(PROG).cpp : 
		
clean:
	rm $(PROG).x
