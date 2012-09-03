#undef _GLIBCXX_ATOMIC_BUILTINS 
#undef _GLIBCXX_USE_INT128 
#include "texArrayHandle.h"
#include "kernel.h"
#include "molecule.h"
#include "parse.h"
#include <map>
#include <utility>
#include <bitset>
#include <math.h>
#include <stdio.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>

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
__device__ __constant__ int dofCount;
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
__device__ __constant__ float cent_coord[3];

__device__ __constant__ int popSize;
__device__ __constant__ int GAIteration;
__device__ __constant__ float mutateRate;
__device__ __constant__ float crossRate;

extern "C" void searchSet(int &pop, int &num_it, float &rate_mutate, float &rate_cross)
{
	cudaMemcpyToSymbol(GAIteration,&num_it,sizeof(int));
	cudaMemcpyToSymbol(mutateRate,&rate_mutate,sizeof(float));
	cudaMemcpyToSymbol(crossRate,&rate_cross,sizeof(float));
	cudaMemcpyToSymbol(popSize,&pop,sizeof(int));
}

extern "C" void mapSet(float &step, float *min, int *dim, float *center)
{
	cudaMemcpyToSymbol(spacing,&step,sizeof(float));
	cudaMemcpyToSymbol(min_coord,min,sizeof(float)*3);
	cudaMemcpyToSymbol(map_dim,dim,sizeof(int)*3);
	cudaMemcpyToSymbol(cent_coord,center,sizeof(float)*3);
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
	t=lig.torCount+7;
	cudaMemcpyToSymbol(dofCount,&t,sizeof(int));

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
		if (((move[id]&(1<<i))!=0)&&(t[i]!=0))
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

__device__ float eIntCalc(int index_dist,int atom1, int atom2, float & local, float & desol)
{
	local=(elec_q[atom1]*elec_q[atom2]*tex1Dfetch(r_eps_ruler,index_dist+0.5f));
	if (index_dist<=800)
	{
		local+=(tex3D(vdw_tables,index_dist+0.5f,type[atom1]+0.5f,type[atom2]+0.5f));
		desol=VOL[type[atom1]]*(SOLPAR[type[atom2]]+0.01097f*fabs(elec_q[atom2]))+VOL[type[atom2]]*(SOLPAR[type[atom1]]+0.01097f*fabs(elec_q[atom1]));
		local+=(tex1Dfetch(sol_ruler,index_dist+0.5f)*desol);
	}
	return local;
}

__device__ float eTrip(int id, float *pos, float & vdw, float & dsol, float & non, float & ele, float & lkid_x, float & lkid_y, float &lkid_z)
{
	if (id>=atomCount || ignore[id])
		return 0;

	vdw=0,dsol=0,non=0,ele=0;

	lkid_x=(pos[0]-min_coord[0])/spacing;
	lkid_y=(pos[1]-min_coord[1])/spacing;
	lkid_z=(pos[2]-min_coord[2])/spacing;

	if (   lkid_x>=0 && lkid_x<map_dim[0]
		&& lkid_y>=0 && lkid_y<map_dim[1]
		&& lkid_z>=0 && lkid_z<map_dim[2])
	{
	arrayTexFetch(type[id],lkid_x+0.5f,lkid_y+0.5f,lkid_z+0.5f,vdw);
	arrayTexFetch(IDX_D,lkid_x+0.5f,lkid_y+0.5f,lkid_z+0.5f,dsol);
	arrayTexFetch(IDX_E,lkid_x+0.5f,lkid_y+0.5f,lkid_z+0.5f,ele);

	non=vdw+dsol*fabs(elec_q[id]);
	ele*=elec_q[id];

	return (non+ele);
	}
	else
		return 1000.0f;
}

__device__ int getDIST(float *now_x, float *now_y, float *now_z, int atom1, int atom2)
{
	return (int)(100.0f*sqrt((now_x[atom1]-now_x[atom2])*(now_x[atom1]-now_x[atom2])+
					(now_y[atom1]-now_y[atom2])*(now_y[atom1]-now_y[atom2])+
					(now_z[atom1]-now_z[atom2])*(now_z[atom1]-now_z[atom2])));
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
		/*printf("%s error quat (%5.3f %5.3f %5.3f %5.3f)\n",msg,vec[3],vec[4],vec[5],vec[6]);*/
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

__device__ void uniformQuat(float * vec, float x,float t1, float t2)
{
	float r=sqrt(x);
	vec[0]=sin(t1)*r;
	vec[1]=cos(t1)*r;
	r=sqrt(1-x);
	vec[2]=sin(t2)*r;
	vec[3]=cos(t2)*r;
	return;
}

__device__ void mutate_quat(float * vec, float x, float t1, float t2)
{
	float quat[4];
	uniformQuat(quat,x,t1,t2);
	qmultiply(vec, quat);
	return;
}

__global__ void init_rng(curandState * const rngStates, const unsigned int seed)
{
	curand_init(seed,blockIdx.x,0,&rngStates[blockIdx.x]);
	/*curand_init(12725,blockIdx.x,0,&rngStates[blockIdx.x]);*/
}

__global__ void init(curandState * const rngStates,float *pool)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x;
	int i=0;
	float x=0.0f,r=0.0f,t=0.0f;
	/*curandState local_state=rngStates[id];*/

if (id<popSize)
{
	for (i=0;i<3;i++)
		pool[i*popSize+id]=curand_normal(&rngStates[id])/3.0f+cent_coord[i];
	x=curand_uniform(&rngStates[id]);
	r=sqrt(x);
	t=curand_uniform(&rngStates[id])*PI2;
	pool[3*popSize+id]=sin(t)*r;
	pool[4*popSize+id]=cos(t)*r;
	t=curand_uniform(&rngStates[id])*PI2;
	r=sqrt(1.0f-x);
	pool[5*popSize+id]=sin(t)*r;
	pool[6*popSize+id]=cos(t)*r;
	for (i=7;i<dofCount;i++)
		pool[i*popSize+id]=curand_uniform(&rngStates[id])*PI-HALFPI;

/*
if(threadIdx.x==0)
{
	pool[0*blockDim.x+id]=2.787401f;
	pool[1*blockDim.x+id]=6.005415f;
	pool[2*blockDim.x+id]=-7.987497f;

	pool[3*blockDim.x+id]=0.499513f;
	pool[4*blockDim.x+id]=-0.515964f;
	pool[5*blockDim.x+id]=-0.340025f;
	pool[6*blockDim.x+id]=0.607166f;

	pool[7*blockDim.x+id]=-1.344776189f;
	pool[8*blockDim.x+id]=-0.3217514477f;
	pool[9*blockDim.x+id]=-0.05846853f;
	pool[10*blockDim.x+id]=-1.4717488921f;
	pool[11*blockDim.x+id]=-0.0616101226f;
	pool[12*blockDim.x+id]=-0.0260926723f;
	pool[13*blockDim.x+id]=1.1554952315f;
	pool[14*blockDim.x+id]=0.7537204376f;
	pool[15*blockDim.x+id]=-0.4110250389f;
	pool[16*blockDim.x+id]=0.7056366167f;
	pool[17*blockDim.x+id]=0.3111049392f;
	pool[18*blockDim.x+id]=0.6558074666f;

}
*/
	/*printf("Individual %3d [ %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %10.3f %10.3f %10.3f ...]\n",id, pool[0*popSize+id], pool[1*popSize+id], pool[2*popSize+id], pool[3*popSize+id], pool[4*popSize+id], pool[5*popSize+id], pool[6*popSize+id], pool[7*popSize+id]*114.591559f, pool[8*popSize+id]*114.591559f, pool[9*popSize+id]*114.591559f);*/

}
	return;
}

__device__ void brute_quat_rot(float x, float y, float z, float w, float * r)
{
	r[0]=1.0f - 2*y*y -2*z*z;
	r[1]=2*x*y-2*w*z;
	r[2]=2*x*z+2*w*y;
	r[3]=2*x*y+2*w*z;
	r[4]=1.0f - 2*x*x -2*z*z;
	r[5]=2*y*z-2*w*x;
	r[6]=2*x*z-2*w*y;
	r[7]=2*y*z+2*w*x;
	r[8]=1.0f - 2*x*x -2*y*y;
}

__global__ void energyEval(int * nonbonded, float * pool, float * score)
{
	__shared__ float now_x[MAX_ATM];
	__shared__ float now_y[MAX_ATM];
	__shared__ float now_z[MAX_ATM];
	__shared__ float energy[MAX_ATM];

	int id=threadIdx.x;
	int i,j;
	float temp[5],fr[9];
if (id<atomCount)
{
	float pos[3]={coord_x[id],coord_y[id],coord_z[id]};
	for (i=0;i<torCount;i++)
	{
		temp[0]=pool[(i+7)*popSize+blockIdx.x];
		if (((move[id]&(1<<i))!=0) && (temp[0]!=0.0f))
		{
			temp[4]=sin(temp[0]);
			temp[1]=temp[4]*axn_x[i];
			temp[2]=temp[4]*axn_y[i];
			temp[3]=temp[4]*axn_z[i];
			temp[4]=cos(temp[0]);
			brute_quat_rot(temp[1],temp[2],temp[3],temp[4],fr);
			
			j=torsion[i][0];
			temp[0]=pos[0]-coord_x[j];
			temp[1]=pos[1]-coord_y[j];
			temp[2]=pos[2]-coord_z[j];

			pos[0]=temp[0]*fr[0]+temp[1]*fr[3]+temp[2]*fr[6]+coord_x[j];
			pos[1]=temp[0]*fr[1]+temp[1]*fr[4]+temp[2]*fr[7]+coord_y[j];
			pos[2]=temp[0]*fr[2]+temp[1]*fr[5]+temp[2]*fr[8]+coord_z[j];
		}
	}
	if (id<ligandCount)
	{
		brute_quat_rot(pool[3*popSize+blockIdx.x],pool[4*popSize+blockIdx.x], pool[5*popSize+blockIdx.x], pool[6*popSize+blockIdx.x],fr);

		temp[0]=pos[0];
		temp[1]=pos[1];
		temp[2]=pos[2];

		pos[0]=temp[0]*fr[0]+temp[1]*fr[3]+temp[2]*fr[6]+pool[0*popSize+blockIdx.x];
		pos[1]=temp[0]*fr[1]+temp[1]*fr[4]+temp[2]*fr[7]+pool[1*popSize+blockIdx.x];
		pos[2]=temp[0]*fr[2]+temp[1]*fr[5]+temp[2]*fr[8]+pool[2*popSize+blockIdx.x];
	}
	now_x[id]=pos[0];
	now_y[id]=pos[1];
	now_z[id]=pos[2];
	__syncthreads();

	temp[0] = eTrip(id, pos, fr[0],fr[1],fr[2],fr[3],fr[4],fr[5],fr[6]);
	temp[1] =0;
	for (i=id+1;i<atomCount;i++)
	{
		if (nonbonded[id*atomCount+i])
		{
			temp[1]+=eIntCalc(getDIST(now_x,now_y,now_z,id,i),id,i,fr[7],fr[8]);
		}
	}
	energy[id]=temp[0]+temp[1];
	__syncthreads();
}
else
{
	energy[id]=0.0f;
}
	/*if (blockIdx.x==0)*/
	/*{*/
		/*printf("atom %3d [ %10.5f %10.5f %10.5f ] %10.5f %10.5f %10.5f\n",id,now_x[id],now_y[id],now_z[id],fr[2],fr[3],temp[1]);*/
		/*printf("atom %2d has energy %10.5f\n",id,energy[id]);*/
	/*}*/

	for (i=blockDim.x/2;i>0;i>>=1)
	{
		if (id < i)
		{
			energy[id]+=energy[id+i];
		}
		__syncthreads();
	}
	if(id==0)
	{
		score[blockIdx.x]=energy[0];
		/*printf("block %2d has score %10.5f\n",blockIdx.x,score[blockIdx.x]);*/
	}
	return;
}

__global__ void energyToScore(float *max, float * sum, float * score)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x;
	if (id<popSize)
	{
		score[id]=(*max-score[id])/(*max-(*sum)/popSize);
		if(score[id]<1.0f) score[id]=0.0f;
		/*printf("score[%2d]=%10.5f\n",id,score[id]);*/
	}
}

__global__ void normalize(float * sum, float *score_acum)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x;
	if (id<popSize)
	{
		/*if (id==0)	printf("\n\n\nsum %f\n",*sum);*/
		score_acum[id]/=(*sum);
	}
}

__global__ void select( float * score, float * poolwork, float *poolbuff, curandState * const rngStates)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x, left=0, right=popSize-1, index=0;
	float dice=0;

if (id<popSize)
{
	dice=curand_uniform(&rngStates[id]);
	while (left!=right && left!=right-1)
	{
		index=(left+right)/2;
		if (score[index]>=dice)
			right=index;
		else left=index;
	}
	index=(score[left]>=dice?left:right);
	/*printf("dice[%2d]=%10.5f\tindex[%2d]=%d\n",id,dice,id,index);*/
	
	// get the offspring, ask Kristin whether memcpy can work here
	for (left=0;left<dofCount;left++)
	{
		poolbuff[left*popSize+id]=poolwork[left*popSize+index];
		__syncthreads();
	}
	/*printf("Individual %3d [ %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %10.3f %10.3f %10.3f ...]\n",id, poolbuff[0*popSize+id], poolbuff[1*popSize+id], poolbuff[2*popSize+id], poolbuff[3*popSize+id], poolbuff[4*popSize+id], poolbuff[5*popSize+id], poolbuff[6*popSize+id], poolbuff[7*popSize+id]*114.591559f, poolbuff[8*popSize+id]*114.591559f, poolbuff[9*popSize+id]*114.591559f);*/
}
}

__device__ float curand_cauchy(curandState * state)
{
	float2 nor_2=curand_normal2(state);
	return nor_2.x/nor_2.y;
}

__global__ void gaOperator(float * pool, curandState * const rngStates)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x;
	float temp, r, t;
	int p1,p2, i;
	float4 t0, t1;

if (id<popSize)
{
	curandState rng=rngStates[id];
	// crossover
	if (id%2==0 && id+1<popSize)
	{
		if (curand_uniform(&rng)<crossRate)
		{
			do
			{
				p1=curand(&rng)%(dofCount-2)+1;
			} while (p1>=4 && p1<=6);
			do
			{
				p2=curand(&rng)%(dofCount-2)+1;
			} while (p1==p2 ||(p2>=4 && p2<=6));

			/*printf("corssing %d-%d @ [ %d %d ]\n",id,id+1, comp_min(p1,p2), comp_max(p1,p2));*/

			for (i=comp_min(p1,p2);i<comp_max(p1,p2);i++)
			{
				temp=pool[i*popSize+id];
				pool[i*popSize+id]=pool[i*popSize+id+1];
				pool[i*popSize+id+1]=temp;
			}
		}
	}
	__syncthreads();

	// mutation
	for (i=0;i<3;i++)
	{
		if (curand_uniform(&rng)<mutateRate)
		{
			/*printf("Mutation! Ind %d Gene %d\n",id,i);*/
			pool[i*popSize+id]+=curand_cauchy(&rng);
		}
	}
	for (i=7;i<dofCount;i++)
	{
		if (curand_uniform(&rng)<mutateRate)
		{
			/*printf("Mutation! Ind %d Gene %d\n",id,i);*/
			temp=curand_cauchy(&rng)*0.5f+pool[i*popSize+id];
			while (temp>=HALFPI) temp-=PI;
			while (temp<-HALFPI) temp+=PI;
			pool[i*popSize+id]=temp;
		}
	}
	if (curand_uniform(&rng)<0.07763184f)
	{
		/*printf("Mutation! Ind %d Gene rot\n",id);*/
		temp=curand_uniform(&rng);
		r=sqrt(temp);
		t=curand_uniform(&rng)*PI2;
		t1.x=sin(t)*r;
		t1.y=cos(t)*r;
		t=curand_uniform(&rng)*PI2;
		r=sqrt(1.0f-temp);
		t1.z=sin(t)*r;
		t1.w=cos(t)*r;
		t0.x=pool[3*popSize+id];
		t0.y=pool[4*popSize+id];
		t0.z=pool[5*popSize+id];
		t0.w=pool[6*popSize+id];
		pool[3*popSize+id]=t0.w*t1.x+t0.x*t1.w+t0.y*t1.z-t0.z*t1.y;
		pool[4*popSize+id]=t0.w*t1.y+t0.y*t1.w+t0.z*t1.x-t0.x*t1.z;
		pool[5*popSize+id]=t0.w*t1.z+t0.z*t1.w+t0.x*t1.y-t0.y*t1.x;
		pool[6*popSize+id]=t0.w*t1.w-t0.x*t1.x-t0.y*t1.y-t0.z*t1.z;
	}

	rngStates[id]=rng;
}
}

__global__ void bestSoFar(float *best_score, float *round_min, float *ga_score, float * best_var, float *pool)
{
	__shared__ int index;
	int id=threadIdx.x;
	int i=0;
	if ((*round_min)<(*best_score))
	{
		if (id==0)
		{
			index=popSize;
			*best_score=*round_min;
		}
		__syncthreads();
		if (ga_score[id]<=(*round_min))
		{
			atomicMin(&index,id);
		}
		__syncthreads();
		if (id==index)
			printf("bestSoFar @ %d\n",id);
		for (i=id;i<dofCount;i+=blockDim.x)
		{
			best_var[i]=pool[i*popSize+index];
		}
		__syncthreads();
		/*if (id==0)	printf("Best (%7.3f) [ %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %10.3f %10.3f %10.3f ...]\n",*best_score, best_var[0], best_var[1], best_var[2], best_var[3], best_var[4], best_var[5], best_var[6], best_var[7]*114.591559f, best_var[8]*114.591559f, best_var[9]*114.591559f);*/
	}
	return;
}

__global__ void elitism(float * pool, curandState * const rngStates, float * best_var)
{
	__shared__ int index;
	int id=threadIdx.x;
	if (id==0)
	{
		index=curand(&rngStates[0])%(popSize);
	}
	__syncthreads();
	pool[id*popSize+index]=best_var[id];
}

__global__ void bestSoFar(int index, float * ga_score, float *best_score, float * best_var, float * pool)
{
	int id=threadIdx.x;
	__shared__ int better;
	if (id==0)
	{
		/*printf("BESTSOFAR @ %d=%f\n",index, ga_score[index]);*/
		if (ga_score[index]<*best_score)
		{
			*best_score=ga_score[index];
			better=1;
		}
		else
			better=0;
	}
	__syncthreads();
	if (better==1)
	{
		/*if (id==0)	printf("BESTSOFAR @ %d\n",index);*/
		best_var[id]=pool[id*popSize+index];
	}
}

__global__ void checkScore(float * score)
{
	printf("score[%d]=%f\n",threadIdx.x,score[threadIdx.x]);
}

extern "C" void launch_kernel(curandState * const rngStates, const unsigned int seed, unsigned int GAround, int sizePop, int sizeAtom, int sizeTor,int * nonbonded, float *pool, float * ga_score, float * h_best_var, float * h_best_score)
{
	float * pool1=0;
	cudaMalloc((void**)&pool1,sizeof(float)*sizePop*(sizeTor+7));
	float * poolwork=pool, *poolbuff=pool1;

	float * d_temp=0;
	cudaMalloc((void**) &d_temp, sizeof(float));
	float *out=0;
	out=(float *)malloc(sizeof(float)*sizePop);
	
	float thrustOut[3];
	float *reduceOut;	// 0:min, 1:max, 2:sum;
	cudaMalloc((void**)&reduceOut,sizeof(float)*3);

	float * best_var=0;
	cudaMalloc((void**)&best_var, sizeof(float)*(sizeTor+7));
	float * best_score=0;
	cudaMalloc((void**)&best_score,sizeof(float));
	float bigFloat=1e9f;
	cudaMemcpy(best_score,&bigFloat,sizeof(float),cudaMemcpyHostToDevice);

	thrust::device_ptr<float> score_ptr(ga_score);
	thrust::device_ptr<float> best_ptr;

	cudaEvent_t start, stop; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	init_rng <<< sizePop, 1 >>> (rngStates, seed);

	int numThreads=256;
	int numBlocks=(sizePop/256)+1;
	printf("Initilization poolwork\n");
	init <<< numBlocks, numThreads >>> (rngStates,poolwork);
	cudaMemcpy(poolbuff,poolwork,sizeof(float)*sizePop*(sizeTor+7),cudaMemcpyDeviceToDevice);

	for (int i=0;i<GAround;i++)
	{
		if (i%2==0)
		{
			poolwork=pool;
			poolbuff=pool1;
		}
		else
		{
			poolwork=pool1;
			poolbuff=pool;
		}
		/*printf("==========Iteration %4d==========\n",i);*/

		// fitness evaluation
		energyEval <<< sizePop, MAX_ATM >>> (nonbonded,poolwork,ga_score);

		// selection
		thrustOut[0]=thrust::reduce(score_ptr,score_ptr+sizePop, 1e9,thrust::minimum<float>());
		thrustOut[1]=thrust::reduce(score_ptr,score_ptr+sizePop,-1e9,thrust::maximum<float>());
		thrustOut[2]=thrust::reduce(score_ptr,score_ptr+sizePop,   0,thrust::plus<float>());
		cudaMemcpy(reduceOut,thrustOut,sizeof(float)*3,cudaMemcpyHostToDevice);

		best_ptr=thrust::min_element(score_ptr,score_ptr+sizePop);
		bestSoFar <<< 1,sizeTor+7 >>> (best_ptr-score_ptr, ga_score,best_score,best_var, poolwork);

		energyToScore <<< numBlocks, numThreads >>> (&reduceOut[1], &reduceOut[2], ga_score);
		thrust::inclusive_scan(score_ptr,score_ptr+sizePop,score_ptr);
		cudaMemcpy(d_temp,ga_score+sizePop-1,sizeof(float),cudaMemcpyDeviceToDevice);
		normalize <<< numBlocks, numThreads >>> (d_temp,ga_score);
		select <<< numBlocks, numThreads >>> (ga_score,poolwork, poolbuff,rngStates);

		// crossover
		// mutation
		gaOperator <<< numBlocks, numThreads >>> (poolbuff,rngStates);
		// elitism
		elitism <<< 1, sizeTor+7 >>> (poolbuff, rngStates, best_var);
		
		/*cudaDeviceSynchronize();*/
		/*printf("\n");*/
	}
	/*energyEval <<< sizePop, MAX_ATM >>> (nonbonded, pool, ga_score);*/
	cudaMemcpy(h_best_score,best_score,sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_best_var, best_var, sizeof(float)*(sizeTor+7), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);

	printf("Kernel total runtime: %f ms\n",elapsedTime);

	free(out);
	cudaFree(best_score);
	cudaFree(best_var);
	cudaFree(d_temp);
	cudaFree(reduceOut);
	cudaFree(pool1);
}

