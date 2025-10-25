//GPU REALISATION CODE

// TYPES SECTOR
struct connection {
	unsigned short b_id;
	unsigned short n_id;
	float init_mult;
	float mult;
	float acc;
	float accoof;
};

struct neuron {
	unsigned short id;
	unsigned char cdata;
	float in_d;
	float state;
	unsigned short c;
};
// Random UINT64 generator
unsigned long random(unsigned long seed) {
	unsigned long result = (seed + 0x9E3779B97F4A7C15);
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);
}

// ACTIVATION FUNCTS SECTOR
static float linear(float x, float c) {
	return 0.5f + x / c / 2.0f;
};

static float softrelu(float x, float c) {
	unsigned char state = ((x / c > 1.0f) << 1) | (x / c < 0.0f);
	float d = 0;
	switch (state) {
	case 0:
		d = x / c;
		break;
	case 1:
		d = x / c / 10.0f;
		break;
	case 2:
		d = 1.0f + (x / c - 1.0f) / 10.0f;
		break;
	}
	return 0.5f + d / 2.0f;
};

static float relu(float x, float c) {
	return 0.5f + min(-1.0f, max(x / c, 1.0f)) / 2.0f;
};

static float sigmoid(float x, float c) {
	return 1.0f/(1.0f+exp(-x/log(c)));
};

// KERNEL SECTOR

//Mutate connections function callback
__kernel void MutateC(__global struct connection* cons, const ulong seed, const float param){
	const unsigned long maxui = 0xFFFFFFFFFFFFFFFF;
	unsigned long j = get_global_id(0);
	unsigned long s_s = seed + (j + j ^ (j << 4)); //pseudo-randomizing seed before random function
	unsigned long rgn1 = random(s_s);
	unsigned long rgn2 = random(rgn1);
	float r1 = ((float)rgn1) / ((float)maxui); //Normalizing to float number from 0.0f to 1.0f
	float r2 = ((float)rgn2) / ((float)maxui); //Normalizing to float number from 0.0f to 1.0f
	float p1 = cons[j].init_mult;
	float p2 = cons[j].accoof;

	cons[j].init_mult += (r1 * 2.0f - 1.0f) * param;
	cons[j].accoof = max(0.0f, min(cons[j].accoof + ((r2 * 2.0f - 1.0f) * param), 1.0f));
	if(isnan(cons[j].init_mult)){
		cons[j].init_mult = p1;
	}
	if (isnan(cons[j].accoof)) {
		cons[j].accoof = p2;
	}
}
// Mutate Neurons function callback
__kernel void MutateN(__global struct neuron* ners, const unsigned long seed, const float param) {
	return;
}

__kernel void Update1(__global struct connection* cons, __global struct neuron* ners) {
	// Step 1 - Set neurons prestates and first update of connections.
	unsigned long i = get_global_id(0);
	struct connection c = cons[i];
	atomic_add(ners[c.n_id], ners[c.b_id].state * c.mult);
	atomic_add(ners[c.n_id].c, 1);
	cons[i].acc = cons[i].acc * cons[i].accoof + ners[c.b_id].state * 0.1f;
};

__kernel void Update2(__global struct neuron* ners) {
	// Step 2 - Update neurons state.
	unsigned short i = get_global_id(0);
	struct neuron n = ners[i];
	if ((n.cdata & 3) == 1) {
		goto end;
	}
	unsigned char tp = (unsigned char)(n.cdata>>2)&15;
	switch (tp) {
	case 0:
		ners[i].state = linear(n.in_d, n.c);
		break;
	case 1:
		ners[i].state = relu(n.in_d, n.c);
		break;
	case 2:
		ners[i].state = softrelu(n.in_d, n.c);
		break;
	case 3:
		ners[i].state = sigmoid(n.in_d, n.c);
		break;
	}
	end:
	ners[i].in_d = 0.0f;
	ners[i].c = 0;
};

__kernel void Update3(__global struct connection* cons, const float g_coof) {
	// Step 3 - Update connections based on good coefficient.
	unsigned long i = get_global_id(0);
	float premul = cons[i].mult + (sign(cons[i].mult) * (cons[i].acc * g_coof));
	float omul = 1.0f + log10(premul) / 2.0f;
	if (!isnan(omul)) {
		cons[i].mult = omul;
	}
}

__kernel void GenRand(__global float* rt, const ulong seed) {
	unsigned long long maxui = 0xFFFFFFFFFFFFFFFF;
	unsigned long long i = get_global_id(0);
	//unsigned long long s_s = seed + (j + j ^ (j << 4)); //pseudo-randomizing seed before random function
	unsigned long long rgn1 = random(seed+i);
	float r1 = ((float)rgn1) / ((float)maxui);//((float)rgn1) / ((float)maxui); //Normalizing to float number from 0.0f to 1.0f
	rt[i] = r1;

}
