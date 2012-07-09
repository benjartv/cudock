#include "texArrayHandle.h"
#include "kernel.h"
#include "molecule.h"
#include "parse.h"
#include <map>
#include <utility>
#include <bitset>
#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

using std::map;
using std::pair;
using std::bitset;

#define MAX_ATM 128
#define MAX_TOR 32
#define MAX_DOF 32

__device__ __constant__ float coord_x[MAX_ATM];
__device__ __constant__ float coord_y[MAX_ATM];
__device__ __constant__ float coord_z[MAX_ATM];
__device__ __constant__ float elec_q[MAX_ATM];
__device__ __constant__ int type[MAX_ATM];
__device__ __constant__ bool ignore[MAX_ATM];
__device__ __constant__ int atomCount;
__device__ __constant__ int flexCount;
__device__ __constant__ int ligandCount;
__device__ __constant__ int torCount;
__device__ __constant__ int torsion[MAX_TOR][2];
__device__ __constant__ unsigned long move[MAX_ATM];
__device__ __constant__ float axn_x[MAX_TOR];
__device__ __constant__ float axn_y[MAX_TOR];
__device__ __constant__ float axn_z[MAX_TOR];


__device__ __constant__ float VOL[27]; // hard-coded here, 27 atom types in total;
__device__ __constant__ float SOLPAR[27]; // hard-coded here, 27 atom types in total;


__device__ __constant__ float min_coord[3];
__device__ __constant__ int map_dim[3];
__device__ __constant__ float spacing;

__device__ __constant__ int popSize;
__device__ __constant__ int GAIteration;
__device__ __constant__ float mutateRate;
__device__ __constant__ float crossRate;

extern "C" void searchSet(int &num_it, float &rate_mutate, float &rate_cross)
{
	cudaMemcpyToSymbol(GAIteration,&num_it,sizeof(int));
	cudaMemcpyToSymbol(mutateRate,&rate_mutate,sizeof(float));
	cudaMemcpyToSymbol(crossRate,&rate_cross,sizeof(float));
	
	// Population size is set to be same as the number of atoms
	//cudaMemcpyToSymbol(popSize,&lig.atomCount,sizeof(int));
}

extern "C" void mapSet(float &step, float *min, int *dim)
{
	cudaMemcpyToSymbol(spacing,&step,sizeof(float));
	cudaMemcpyToSymbol(min_coord,min,sizeof(float)*3);
	cudaMemcpyToSymbol(map_dim,dim,sizeof(int)*3);
}

extern void init_texture(tex3DHandle<float> &handle, int index,bool filtered)
{
	const textureReference * texPtr;
	if      (index==IDX_E) cudaGetTextureReference(&texPtr, "tex_e");
	else if (index==IDX_D) cudaGetTextureReference(&texPtr, "tex_d");
	else if (index==IDX_VDW) cudaGetTextureReference(&texPtr, "vdw_tables");
	else if ((index>=0)&&(index<=26))cudaGetTextureReference(&texPtr, texName[index]);
	textureReference * texRefPtr=const_cast<textureReference *> (texPtr);
	texRefPtr->normalized=false;
	if (filtered)
		texRefPtr->filterMode=cudaFilterModeLinear;
	else
		texRefPtr->filterMode=cudaFilterModePoint;

	texRefPtr->addressMode[0]=cudaAddressModeClamp;
	texRefPtr->addressMode[1]=cudaAddressModeClamp;
	texRefPtr->addressMode[2]=cudaAddressModeClamp;

	cudaBindTextureToArray(texPtr,handle.getPtr(),handle.getChannel());
}

extern void init_texture(tex1DHandle<float> &handle, int index)
{
	const textureReference * texPtr;
	if (index==IDX_SOL) cudaGetTextureReference(&texPtr, "sol_ruler");
	else if (index==IDX_EPS) cudaGetTextureReference(&texPtr, "eps_ruler");
	else if (index==IDX_R_EPS) cudaGetTextureReference(&texPtr, "r_eps_ruler");
	textureReference * texRefPtr=const_cast<textureReference *> (texPtr);
	texRefPtr->normalized=false;
	texRefPtr->filterMode=cudaFilterModePoint;
	texRefPtr->addressMode[0]=cudaAddressModeClamp;

	size_t offset=0;

	cudaBindTexture(&offset,texPtr,(void*)handle.getPtr(),handle.getChannel(),1000*sizeof(float));
}

extern "C" void ligandCopy(ligand &lig)
{
	cudaMemcpyToSymbol(VOL,&vol,sizeof(float)*27);
	cudaMemcpyToSymbol(SOLPAR,&solpar,sizeof(float)*27);

	int n=lig.atomCount;
	int t=n-lig.flexCount;
	cudaMemcpyToSymbol(atomCount,&lig.atomCount,sizeof(int));
	int pop=MAX_ATM;
	cudaMemcpyToSymbol(popSize,&pop,sizeof(int));

	cudaMemcpyToSymbol(flexCount,&lig.flexCount,sizeof(int));
	cudaMemcpyToSymbol(ligandCount,&t,sizeof(int));
	cudaMemcpyToSymbol(coord_x,&lig.coord_x[0],sizeof(float)*n);
	cudaMemcpyToSymbol(coord_y,&lig.coord_y[0],sizeof(float)*n);
	cudaMemcpyToSymbol(coord_z,&lig.coord_z[0],sizeof(float)*n);
	cudaMemcpyToSymbol(elec_q, &lig.elecq[0]  ,sizeof(float)*n);
	cudaMemcpyToSymbol(type,   &lig.type[0]   ,sizeof(float)*n);

	t=lig.torCount;
	cudaMemcpyToSymbol(torCount,&t,sizeof(int));

	int axi[t][2];
	unsigned long moving[n];
	bitset<MAX_ATM> bitTor[n];
	bool init_ignore[n];

	float h_axn_x[t];
	float h_axn_y[t];
	float h_axn_z[t];


	for (int i=0;i<n;i++)
	{
		bitTor[i].reset();
		moving[i]=0;
		if (lig.ignore.test(i))
			init_ignore[i]=true;
		else init_ignore[i]=false;
	}

	cudaMemcpyToSymbol(ignore, init_ignore ,sizeof(bool)*n);

	
	map < pair<int, int>, bitset<MAX_ATM> >:: iterator it;


	for (int p=0;p<lig.torCount;p++)
	{
		axi[p][0]=lig.base[p].first;
		axi[p][1]=lig.base[p].second;

		for (int i=0;i<n;i++)
		{
			if (lig.torsion[p].test(i))
				bitTor[i].set(p);
		}

		h_axn_x[p]=(lig.coord_x[axi[p][0]]-lig.coord_x[axi[p][1]]);
		h_axn_y[p]=(lig.coord_y[axi[p][0]]-lig.coord_y[axi[p][1]]);
		h_axn_z[p]=(lig.coord_z[axi[p][0]]-lig.coord_z[axi[p][1]]);
		float dist=sqrt(h_axn_x[p]*h_axn_x[p]+h_axn_y[p]*h_axn_y[p]+h_axn_z[p]*h_axn_z[p]);
		h_axn_x[p]/=dist;
		h_axn_y[p]/=dist;
		h_axn_z[p]/=dist;

	}

	for (int i=0;i<n;i++)
	{
		moving[i]=bitTor[i].to_ulong();
	}

	cudaMemcpyToSymbol(torsion,axi,sizeof(float)*2*t);
	cudaMemcpyToSymbol(move,moving,sizeof(unsigned long)*n);
	cudaMemcpyToSymbol(axn_x,h_axn_x,sizeof(float)*t);
	cudaMemcpyToSymbol(axn_y,h_axn_y,sizeof(float)*t);
	cudaMemcpyToSymbol(axn_z,h_axn_z,sizeof(float)*t);
}

__device__ void quat_rot(float x, float y, float z, float w, float *r)
{
	float tx=x+x;
	float ty=y+y;
	float tz=z+z;
	float twx=w*tx;
	float omtxx=1.0-x*tx;
	float txy=y*tx;
	float txz=z*tx;
	float twy=w*ty;
	float tyy=y*ty;
	float tyz=z*ty;
	float twz=w*tz;
	float tzz=z*tz;

	r[0]=1.0-tyy-tzz;
	r[1]=txy-twz;
	r[2]=txz+twy;
	r[3]=txy+twz;
	r[4]=omtxx-tzz;
	r[5]=tyz-twx;
	r[6]=txz-twy;
	r[7]=tyz+twx;
	r[8]=omtxx-tyy;
}

__device__ void transform(float *out, float x, float y, float z, float * fr)
{
	// index is the index of the atom
	// (x,y,z) is the translation vector
	// fr is the rotational factor vector
	// out(x,y,z) is the coordinates of output

	float cdx=out[0],cdy=out[1],cdz=out[2];

	out[0]=cdx*fr[0]+cdy*fr[3]+cdz*fr[6]+x;
	out[1]=cdx*fr[1]+cdy*fr[4]+cdz*fr[7]+y;
	out[2]=cdx*fr[2]+cdy*fr[5]+cdz*fr[8]+z;

	return;
}

__device__ void twist(float *out, int id, float *t)
// make sure twist is done before any transform
// twist is on the original coordinates
{
	for (int i=0;i<torCount;i++)
	{
		if (((move[id]&(1<<i))!=0)&&(t[i]!=0.0f))
		{
			float theta_half=t[i];
			float a=sin(theta_half)*axn_x[i];
			float b=sin(theta_half)*axn_y[i];
			float c=sin(theta_half)*axn_z[i];
			float w=cos(theta_half);
			float fr[9];
			quat_rot(a,b,c,w,fr);
			int ori=torsion[i][0];
			float ori_x=coord_x[ori];
			float ori_y=coord_y[ori];
			float ori_z=coord_z[ori];
			out[0]-=ori_x;
			out[1]-=ori_y;
			out[2]-=ori_z;
			transform(out,ori_x,ori_y,ori_z,fr);
		}
	}
}

__device__ float eIntCalc(int index_dist,int atom1, int atom2)
{
	float local=0;
	local+=(elec_q[atom1]*elec_q[atom2]*tex1Dfetch(r_eps_ruler,index_dist+0.5f));
	if (index_dist<=800)
	{
		local+=(tex3D(vdw_tables,index_dist+0.5f,type[atom1]+0.5f,type[atom2]+0.5f));
		float desol=VOL[type[atom1]]*(SOLPAR[type[atom2]]+0.01097f*fabs(elec_q[atom2]))+VOL[type[atom2]]*(SOLPAR[type[atom1]]+0.01097f*fabs(elec_q[atom1]));
		local+=(tex1Dfetch(sol_ruler,index_dist+0.5f)*desol);
	}
	return local;
}

__device__ float eTrip(int id, float *pos)
{
	if (ignore[id])
		return 0;

	float vdw=0,dsol=0,non=0,ele=0;

	float lkid_x=(pos[0]-min_coord[0])/spacing;
	float lkid_y=(pos[1]-min_coord[1])/spacing;
	float lkid_z=(pos[2]-min_coord[2])/spacing;

	if (   lkid_x>=0 && lkid_x<61
		&& lkid_y>=0 && lkid_y<61
		&& lkid_z>=0 && lkid_z<67)
	{
	arrayTexFetch(type[id],lkid_x+0.5f,lkid_y+0.5f,lkid_z+0.5f,vdw);
	arrayTexFetch(IDX_D,lkid_x+0.5f,lkid_y+0.5f,lkid_z+0.5f,dsol);
	arrayTexFetch(IDX_E,lkid_x+0.5f,lkid_y+0.5f,lkid_z+0.5f,ele);

	non=vdw+dsol*abs(elec_q[id]);
	ele*=elec_q[id];

	return (non+ele);
	}
	else
		return 1000.0f;
}

__device__ int getDIST(float *now_x, float *now_y, float *now_z, int atom1, int atom2)
{
	float dist=sqrt((now_x[atom1]-now_x[atom2])*(now_x[atom1]-now_x[atom2])+
					(now_y[atom1]-now_y[atom2])*(now_y[atom1]-now_y[atom2])+
					(now_z[atom1]-now_z[atom2])*(now_z[atom1]-now_z[atom2]));
	return (int)(dist*100);
}

__device__ void blockEnergy(int id, float *var, float *now_x, float *now_y, float *now_z, float *energy, short *nonbonded, float *local)
// using a entire block to get one conformation scored.
// now_x[], now_y[], now_z[] and energy[] should be shared within the block
// block size == atomCount, so one atom is assigned to one thread;
// so thread syncing might be a performance bottleneck, as might be the uneven computation tasks
{
if (id<atomCount)
{
	float pos[3]={local[0],local[1],local[2]}; // seems could be coalesced

	float *tor_angle=var+7; // careful not to get to null addresses
	twist(pos,id,tor_angle);

	float fr[9];
	quat_rot(var[3],var[4],var[5],var[6],fr);
	
	if (id<(ligandCount))
		transform(pos,var[0],var[1],var[2],fr);
	now_x[id]=pos[0];
	now_y[id]=pos[1];
	now_z[id]=pos[2];

	float interE=eTrip(id,pos);
	__syncthreads();

	float intraE=0;
	for (int i=id+1;i<atomCount;i++)
	{
		if (nonbonded[id*atomCount+i])
		{
			intraE+=eIntCalc(getDIST(now_x,now_y,now_z,id,i),id,i);
		}
	}
	energy[id]=interE+intraE;
}
else
{
	energy[id]=0.0f;
}
__syncthreads();
return;
}

__device__ void normalize(float * vec)
{
	float tr=sin(vec[3])/(sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]));
	vec[3]=cos(vec[3]);
	vec[0]*=tr;
	vec[1]*=tr;
	vec[2]*=tr;
}

__device__ bool validate(float *vec, char * msg)
{
	float temp=vec[3]*vec[3]+vec[4]*vec[4]+vec[5]*vec[5]+vec[6]*vec[6];
	if (temp<0.998f || temp >1.002f)
	{
		printf("%s error quat (%5.3f %5.3f %5.3f %5.3f)\n",msg,vec[3],vec[4],vec[5],vec[6]);
		return false;
	}

	while (vec[0]<-2.5f) vec[0]+=10.0f;
	while (vec[1]< 1.0f) vec[1]+=10.0f;
	while (vec[2]<-13.0f)vec[2]+=10.0f;

	while (vec[0]>=7.5f) vec[0]-=10.0f;
	while (vec[1]>=11.0f)vec[1]-=10.0f;
	while (vec[2]>=-3.0f) vec[2]-=10.0f;



	for (int i=0;i<torCount;++i)
	{
		while (vec[i+7]>=HALFPI) vec[i+7]-=PI;
		while (vec[i+7]<-HALFPI) vec[i+7]+=PI;
	}

	return true;
}

__device__ void qmultiply(float *left, float *right)
{
	float x,y,z,w;
	x=(float)(left[3]*right[0]+left[0]*right[3]+left[1]*right[2]-left[2]*right[1]);
	y=(float)(left[3]*right[1]+left[1]*right[3]+left[2]*right[0]-left[0]*right[2]);
	z=(float)(left[3]*right[2]+left[2]*right[3]+left[0]*right[1]-left[1]*right[0]);
	w=(float)(left[3]*right[3]-left[0]*right[0]-left[1]*right[1]-left[2]*right[2]);
	left[0]=x;
	left[1]=y;
	left[2]=z;
	left[3]=w;
}

__device__ void mutate_quat(float * vec, float x, float t1, float t2)
{
	float quat[4];
	float x0=sqrt(x);
	float x1=sqrt(1-x);
	quat[0]=sin(t1)*(x0);
	quat[1]=cos(t1)*(x0);
	quat[2]=sin(t2)*x1;
	quat[3]=cos(t2)*x1;

	qmultiply(vec, quat);
}

__global__ void init(curandState * const rngStates,const unsigned int seed, float *bridge)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(seed,tid,0,&rngStates[tid]);

	int bridgeOffset=torCount+7;
	float pool[MAX_DOF];
	for (int i=0;i<MAX_DOF;i++) pool[i]=0.0f;
	curandState local_state=rngStates[tid];

	pool[0]=curand_normal(&local_state)/3.0+2.5435f;	// (x, y, z)
	pool[1]=curand_normal(&local_state)/3.0+6.02255f;
	pool[2]=curand_normal(&local_state)/3.0-7.96630f;

	pool[3]=curand_uniform(&local_state)-0.5f;
	pool[4]=curand_uniform(&local_state)-0.5f;
	pool[5]=curand_uniform(&local_state)-0.5f;
	pool[6]=curand_uniform(&local_state)*PI-HALFPI;	// pass half of the angle to get quaternion
	normalize(&pool[3]);

	for (int i=0;i<torCount;++i)
		pool[7+i]=curand_uniform(&local_state)*PI-HALFPI;	// half of the angle
	if (validate(pool,"init"))
	{
		for (int i=0;i<torCount+7;i++)
			bridge[tid*bridgeOffset+i]=pool[i];
	}
	if (validate(&bridge[tid*bridgeOffset],"bridge"))
	{
		/*printf("bridge[%d] is OK\n",tid);*/
	}
	rngStates[tid]=local_state;

	return;
}


__global__ void tryGA(curandState * const rngStates, short *nonbonded, float * bridge, float * out_score, float * out_var)
{
	// bridge is with size (torCount+7)*blockSize*gridSize: bridge[DOF][blockIdx*blockDim+threadIdx]
	// bridge[DOFIdx*gridSize*blockSize+blockIdx*blockDim+threadIdx]
	__shared__ float now_x[MAX_ATM];
	__shared__ float now_y[MAX_ATM];
	__shared__ float now_z[MAX_ATM];
	__shared__ float energy[MAX_ATM];
	/*__shared__ float pool[MAX_ATM][MAX_DOF]; // pool[PopSize][DegreeOfFreedom]*/
	__shared__ float score[MAX_ATM]; // set popSize=atomCount
	__shared__ float round_min, round_max, sum, avg;
	__shared__ float blockMin;
	__shared__ float blockOut[MAX_DOF]; // DegreeOfFreedom=32

	float offspring[MAX_DOF];
	int id=threadIdx.x;
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int bridgeOffset=torCount+7;
	curandState local_state=rngStates[tid];

	/*int bridgeOffset=gridDim.x*blockDim.x;*/
	/*int migID=((blockIdx.x+1)%gridDim.x)*blockDim.x+threadIdx.x;*/
	/*float varBuff[MAX_DOF];*/

	float local[3];

	if (id<atomCount)
	{
		local[0]=coord_x[id];
		local[1]=coord_y[id];
		local[2]=coord_z[id];
	}
	else
	{
		local[0]=local[1]=local[2]=0.0f;
	}


	/*printf("block %3d\tthread %3d\ttid %3d\tmigID %3d\n",blockIdx.x,threadIdx.x,tid,migID);*/

	// init population
	score[id]=0.0f;	// the score

	/*pool[id][0]=curand_normal(&local_state)/3.0+2.5435f;	// (x, y, z)*/
	/*pool[id][1]=curand_normal(&local_state)/3.0+6.02255f;*/
	/*pool[id][2]=curand_normal(&local_state)/3.0-7.96630f;*/

	/*pool[id][3]=curand_uniform(&local_state)-0.5f;*/
	/*pool[id][4]=curand_uniform(&local_state)-0.5f;*/
	/*pool[id][5]=curand_uniform(&local_state)-0.5f;*/
	/*pool[id][6]=curand_uniform(&local_state)*PI-HALFPI;	// pass half of the angle to get quaternion*/
	/*normalize(&pool[id][3]);*/

	/*for (int i=0;i<torCount;++i)*/
		/*pool[id][7+i]=curand_uniform(&local_state)*PI-HALFPI;	// half of the angle*/

	__syncthreads();

	if(id==0)
	{
		blockMin=1e9f;
		/*for (int i=0;i<torCount+7;i++)*/
			/*blockOut[i]=pool[0][i];*/
	}

	/*for (int i=0;i<torCount+7;++i)*/
	/*{*/
		/*bridge[i*bridgeOffset+tid]=pool[id][i];*/
	/*}*/
	/*__syncthreads();*/
	
	/*validate(pool[id],"initial");*/

	__syncthreads();
//============================================================================================================
	for (int iteration=0;iteration<GAIteration;iteration++)
	/*for (int iteration=0;iteration<10;iteration++)*/
	{
		if(id==0)
		{
			round_min=1e9, round_max=-1e9;
			sum=0, avg=0;
		}
		__syncthreads();

		//migration
		if (curand_uniform(&local_state)<0.01f)
		{
			for (int i=0;i<torCount+7;i++)
			{
				offspring[i]=bridge[(((blockIdx.x+1)*blockDim.x)%gridDim.x+id)*bridgeOffset+i];
			}
			if (validate(offspring, "migration"))
			{
				for (int i=0;i<torCount+7;i++)
				bridge[tid*bridgeOffset+i]=offspring[i];
			}
		}

		__syncthreads();

		// fitness evaluation
		for (int chromID=0;chromID<popSize;chromID++)
		{
			__syncthreads();
			blockEnergy(id,&bridge[(blockIdx.x*blockDim.x+chromID)*bridgeOffset],now_x,now_y,now_z,energy,nonbonded,local);
			__syncthreads();
			if (id==0)
			{
				float total=0.0f;
				for (int i=0;i<atomCount;i++)
				{
					total+=energy[i];
				}	// !!! should be a reduction here
				if (total<round_min) round_min=total;
				if (total>round_max) round_max=total;

				if (total<blockMin) // this maybe should be done after chromID iteration
				{
					blockMin=total;
					for (int i=0;i<torCount+7;i++)
						blockOut[i]=bridge[(blockIdx.x*blockDim.x+chromID)*bridgeOffset+i];	// ask Kristin whethere there's something like memcpy() in kernel
					// printf("total %10.4f\n",total);
				}
				score[chromID]=total;
				avg+=total;
				/*printf("iter %3d block %3d ID %3d score %3f\n",iteration, blockIdx.x, chromID, score[chromID]);*/
			}
			__syncthreads();
		}
		if(id==0) avg/=(float)popSize;
		__syncthreads();

		//selection
		// for id = 0 to popSize(popSize=atomCount)
			score[id]=(round_max-score[id])/(round_max-avg); // this line and the following line
			if(score[id]<1) score[id]=0;	// is to eliminate the down average individules
			__syncthreads();

			if (id==0)	// !!! should be a reduction here
			{
				for (int i=1;i<popSize;i++)
					score[i]+=score[i-1];
				sum=score[popSize-1];
			}
			__syncthreads();

			score[id]/=sum;
			__syncthreads();

			float dice=curand_uniform(&local_state);
			int index=0;
			while (index<popSize && score[index]<dice) index++;	// !!! should be a binary search
			if (index>=popSize) index=popSize-1;
			for (int i=0;i<torCount+7;i++)	// memcpy()?
				offspring[i]=bridge[(blockIdx.x*blockDim.x+index)*bridgeOffset+i];
			__syncthreads();

			for (int i=0;i<torCount+7;i++)
				bridge[tid*bridgeOffset+i]=offspring[i];	// memcpy()?

			validate(&bridge[tid*bridgeOffset],"selection");


		__syncthreads();

		// cross over
		if ((id%2==0) && (id+1<popSize))
		{
			if (curand_uniform(&local_state)<0.8f)
			{
				float temp;
				int p1,p2;
				do
				{
					p1=(int)(curand_uniform(&local_state)*(torCount+7-1)+1.0f);
				} while (p1==4 || p1==5 || p1==6);
				do
				{
					p2=(int)(curand_uniform(&local_state)*(torCount+7-1)+1.0f);
				} while (p2==4 || p2==5 || p2==6 || p2==p1);

				for (int i=comp_min(p1,p2);i<comp_max(p1,p2);i++)
				{
					temp=bridge[tid*bridgeOffset+i];
					bridge[tid*bridgeOffset+i]=bridge[(tid+1)*bridgeOffset+i];
					bridge[(tid+1)*bridgeOffset+i]=temp;
				}
			}
		}
			validate(&bridge[tid*bridgeOffset],"cross");
		__syncthreads();

		//mutation
		// for id = 0 to popSize(popSize=atomCount)
			for (int i=0;i<torCount+7;i++)
			{
				if (curand_uniform(&local_state)<0.02f)
				{
					if (i>=3 && i<7)
					{
						mutate_quat(&bridge[tid*bridgeOffset+3],curand_uniform(&local_state),curand_uniform(&local_state)*PI2,curand_uniform(&local_state)*PI2);
						i=6;
					}
					else
					{
						bridge[tid*bridgeOffset+i]+=(float)((curand_normal(&local_state))/(curand_normal(&local_state)));
					}
				}
			}
			validate(&bridge[tid*bridgeOffset],"mutate");
		__syncthreads();

		//elitism
		if(id==0)
		{
			for (int i=0;i<torCount+7;i++)
				bridge[tid*bridgeOffset+i]=blockOut[i];
		}

		__syncthreads();

	} //GAIteration

	__syncthreads();

	/*if (id<torCount+7)*/
		/*out_var[blockIdx.x*(torCount+7)+id]=blockOut[id];*/

	if (id==0)
	{
		for (int i=0;i<torCount+7;i++)
			out_var[blockIdx.x*(torCount+7)+i]=blockOut[i];
	}

	blockEnergy(id,blockOut,now_x,now_y,now_z,energy,nonbonded,local);
	/*if (id<atomCount)*/
		/*printf("atom %d [ %10.3f %10.3f %10.3f ] %10.5f\n",id+1, now_x[id], now_y[id], now_z[id], energy[id]);*/
	__syncthreads();
		if (id==0)
		{
				float total=0;
				for (int i=0;i<atomCount;i++)
				{
					total+=energy[i];
				}
				out_score[blockIdx.x]=total;
				printf("block %d Min %10.5f\n",blockIdx.x, total);
		}
	rngStates[tid]=local_state;
}


extern "C" void launch_kernel(curandState * const rngStates, const unsigned int seed,int gridSize, int blockSize,short * nonbonded, float *bridge, float * out_score, float * out_var)
{
	cudaEvent_t start, stop; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	init <<< gridSize, blockSize >>> (rngStates,seed,bridge);

	printf("Initilization done\n");

	tryGA <<< gridSize, blockSize >>> (rngStates, nonbonded,bridge, out_score, out_var);

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);

	printf("Kernel total runtime: %f ms\n",elapsedTime);
}

