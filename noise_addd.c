#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VECTOR_SIZE 10
//			    0b       /       /       /       /
#define unitone 0b00000000000000000000000000000001
#define unitzro 0b00000000000000000000000000000000

typedef union {
    float f;
    unsigned u;
} FloatUnion;

unsigned floatToBinary(float num) {
    FloatUnion fu;
    fu.f = num;
    return fu.u;
	
}

float binaryToFloat(unsigned bin) {
    FloatUnion fu;
    fu.u = bin;
    return fu.f;
}

/* 
float addNoiseToBinary(float number, float errorProbability) {
    unsigned binaryValue = floatToBinary(number);
	FloatUnion res;
	res.u=unitzro;
	for (int i = 0; i < 23; i++) {
		res.u = res.u<<1;
		float randomValue = ((float)rand() / RAND_MAX);  // Random value between 0 and 1
		if (randomValue < (errorProbability)) {
                res.u = res.u ^ unitone;
            }
	}
	printf("ori:");
	for (int i = 31; i >= 0; i--){
		int bit = (binaryValue >> i) & 1;
		printf("%d",bit);
	}
	printf("\n");
	printf("err:");
	for (int i = 31; i >= 0; i--){
		int bit = (res.u >> i) & 1;
		printf("%d",bit);
	}
	printf("\n");
	printf("res:");
	res.u = res.u ^ binaryValue;
	for (int i = 31; i >= 0; i--){
		int bit = (res.u >> i) & 1;
		printf("%d",bit);
	}
	printf("\n");
	printf("\n");
	return res.f;
}
 */

float addNoiseToBinary(float number, float errorProbability) {
    unsigned binaryValue = floatToBinary(number);
	FloatUnion res;
	res.u=unitzro;
	for (int i = 0; i < 23; i++) {
		res.u = res.u<<1;
		float randomValue = ((float)rand() / RAND_MAX);  // Random value between 0 and 1
		if (randomValue < (errorProbability)) {
                res.u = res.u ^ unitone;
            }
	}
	res.u = res.u ^ binaryValue;
	return res.f;
}

float* parallelProcess(float* flatVector, int vectorSize, float errorRate) {
    for (int i = 0; i < vectorSize; ++i) {
        float current_value = flatVector[i];
        flatVector[i] = addNoiseToBinary(current_value, errorRate);
        //printf("%f",flatVector[i]);
    }

    return flatVector;
}