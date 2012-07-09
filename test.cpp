#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "parse.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "texArrayHandle.h"
#include <vector>
#include "molecule.h"
#include "energy_ruler.h"
#include <sys/types.h>

using std::vector;

extern const int IDX_E;
extern const int IDX_D;
extern const int IDX_VDW;
extern const int IDX_SOL;
extern const int IDX_EPS;
extern const int IDX_R_EPS;

extern   void init_texture(tex3DHandle<float>&, int, bool);
extern   void init_texture(tex1DHandle<float>&, int);
extern "C" void launch_kernel(curandState * const, const unsigned int, unsigned int, int,int, int, int *,float *, float *, float *, float *);
extern "C" void ligandCopy(ligand &);
extern "C" void mapSet(float &, float *, int *i, float *);
extern "C" void searchSet(int &, int &, float &, float &);

int main(int argc, char * argv[])
{
	// SCAN DPF
	if (argc!=4)
	{
		printf("Please specify a valid DPF file\n");
		return 1;
	}
	int testSize=1;
	sscanf(argv[1],"%d",&testSize);

	int testRound=1;
	sscanf(argv[2],"%d",&testRound);

	FILE * DPF = fopen(argv[3],"r");
	if (DPF==NULL)
	{
		printf("Please specify a valid DPF file\n");
		return 1;
	}

	char line[1000], temp[1000], var[1000];
	char lig_name[1000], flex_name[1000]={};
	int dim[3],mapCount;
	char mapName[1000][50];
	int skip[50];
	float spacing=0;
	float max_coord[3];
	float min_coord[3];
	float center[3];

	float rate_mutate=0.02, rate_cross=0.8;
	int max_round=1000;
	
	while (fgets(line,1000,DPF)!=NULL)
	{
		int token=parse_dpf(line);
		switch (token)
		{
			case DPF_fld:
			{
				sscanf(line,"%s %s", temp, var);
				FILE * FLD = fopen(var,"r");
				FILE * XYZ;

				// LOAD MAPS INFO
				/*
				   1. scan maps.fld for dims, map names and line skipping
			   	   2. scan maps.xyz for the boundary
				*/
				while (fgets(line,1000,FLD)!=NULL)
				{
					if (line[0]=='#') continue;
					if (strncmp(line,"dim1",4)==0)
					{
						sscanf(line,"%4s=%d",temp,&dim[0]);
					}
					else if (strncmp(line,"dim2",4)==0)
					{
						sscanf(line,"%4s=%d",temp,&dim[1]);
					}
					else if (strncmp(line,"dim3",4)==0)
					{
						sscanf(line,"%4s=%d",temp,&dim[2]);
					}
					else if (strncmp(line,"veclen",6)==0)
					{
						sscanf(line,"%6s=%d",temp,&mapCount);
					}
					else if (strncmp(line,"coord 1",7)==0)
					{
						sscanf(line,"%s %s %4s=%s ",temp,temp,temp,var);
					}
					else if (strncmp(line,"variable",8)==0)
					{
						int index;
						sscanf(line,"%s %d",temp,&index);
						sscanf(line,"%s %s %4s=%s %s %4s=%d",temp,temp,temp,mapName[index-1],temp,temp,&skip[index-1]);
					}
				}
				
				XYZ=fopen(var,"r");
				for (int i=0;i<3;i++)
				{
					fscanf(XYZ,"%f %f",&min_coord[i],&max_coord[i]);
					center[i]=((min_coord[i]+max_coord[i])/2.0);
				}

				fclose(XYZ);


				fclose(FLD);
				break;
			}
			case DPF_map:
				break;
			case DPF_move:
				sscanf(line,"%*s %s\n",lig_name);
				break;
			case DPF_flex:
				sscanf(line,"%*s %s\n",flex_name);
				break;
			default:
				break;
		}
	}

	fclose(DPF);
	// END SCAN DPF
	spacing=(max_coord[0]-min_coord[0])/((float)dim[0]-1);

	//printf("Grid configurations:\n");
	//printf("center\t[%f %f %f]\n",center[0],center[1],center[2]);
	//printf("min\t[%f %f %f]\n",min_coord[0],min_coord[1],min_coord[2]);
	//printf("max\t[%f %f %f]\n",max_coord[0],max_coord[1],max_coord[2]);
	//printf("dims\t[%d %d %d]\n",dim[0],dim[1],dim[2]);
	//printf("spacing\t%.4f\n\n",spacing);

	mapSet(spacing,min_coord,dim,center);

	// LOADING MAPS
	FILE * MAP;
	float * maps=(float * )malloc(sizeof(float)*mapCount*dim[2]*dim[1]*dim[0]);
	tex3DHandle<float> * tex[mapCount];

	for (int i=0;i<mapCount;i++)
	{
		//printf("Map #%d: %s\n",i,mapName[i]);
		MAP=fopen(mapName[i],"r");
		for (int j=0;j<skip[i];j++)
			fgets(line,1000,MAP);
		for (int z=0;z<dim[2];z++)
			for (int y=0;y<dim[1];y++)
				for (int x=0;x<dim[0];x++)
				{
					fscanf(MAP,"%f",&maps[i*dim[2]*dim[1]*dim[0]+z*dim[1]*dim[0]+y*dim[0]+x]);
				}
		fclose(MAP);

		tex[i]= new tex3DHandle<float>(dim[0],dim[1],dim[2]);
		(*tex[i]).copyData((void*)(maps+i*dim[2]*dim[1]*dim[0]));
		init_texture(*tex[i],getIDX(mapName[i]),true);
	}

	//printf("\n");

	// LOADING ligand
	//
	
	int atomCount=0;
	int torCount=0;
	ligand moving;

	if (strlen(lig_name)>0)
	{
		if (strlen(flex_name)>0)
		{
			ligand temp(lig_name,flex_name);
			moving=temp;
		}
		else
		{
			ligand temp(lig_name);
			moving=temp;
		}
	}
		printf("total\n");
		moving.print_tor();
		//moving.print_atom();
		ligandCopy(moving);
		atomCount=moving.atomCount;
		torCount=moving.torCount;
		
		energyRuler ruler;

		tex3DHandle<float> * vdw_map=new tex3DHandle<float>(1000,27,27);
		//tex3DHandle<float> vdw_map(1000,27,27);
		
		(*vdw_map).copyData((void*)(ruler.vdw_table));
		init_texture(*vdw_map,IDX_VDW,false);

		tex1DHandle<float> * desol=new tex1DHandle<float>(1000);
		(*desol).copyData((void*)(ruler.sol_table));
		init_texture(*desol,IDX_SOL);

		tex1DHandle<float> * epsilon=new tex1DHandle<float>(1000);
		(*epsilon).copyData((void*)(ruler.epsilon));
		init_texture(*epsilon,IDX_EPS);

		tex1DHandle<float> * r_eps=new tex1DHandle<float>(1000);
		(*r_eps).copyData((void*)(ruler.r_epsilon));
		init_texture(*r_eps,IDX_R_EPS);
	
	int popSize=testSize;

	searchSet(popSize,max_round,rate_mutate, rate_cross);

	printf("popSize=%5d\tatomCount=%5d\tdofCount=%5d\n",popSize,atomCount,torCount+7);

	curandState * d_rngStates=0;
	cudaMalloc((void**)&d_rngStates,popSize*sizeof(curandState));

	float *pool=0;
#define MAX_DOF 32
	cudaMalloc((void**)&pool,popSize*(torCount+7)*sizeof(float));
	float *ga_score=0;
	cudaMalloc((void**)&ga_score,popSize*sizeof(float));

	pid_t pid=getpid();
	printf("seed %d\n",pid);

	int * nonBond;
	cudaMalloc((void**)&nonBond,sizeof(int)*atomCount*atomCount);
	int tempMatrix[atomCount*atomCount];
	for (int i=0;i<atomCount;++i)
		for (int j=0;j<atomCount;++j)
		{
			tempMatrix[i*atomCount+j]=moving.nonbonded[i][j];
		}
	cudaMemcpy(nonBond,tempMatrix,sizeof(int)*atomCount*atomCount,cudaMemcpyHostToDevice);

	unsigned int round=testRound;
	
	float * best_score=(float*)malloc(sizeof(float));
	float * best_var=(float*)malloc(sizeof(float)*(torCount+7));

	launch_kernel(d_rngStates, (unsigned int)pid,round, popSize, atomCount, torCount, nonBond, pool, ga_score, best_var, best_score);

	printf("Final result\nScore=%10.3f\n",*best_score);
	for (int j=0;j<7;++j)
		printf("%6.3f ",best_var[j]);
	for (int j=7;j<7+torCount;++j)
		printf("%8.3f ",best_var[j]*2.0f*57.29577951f);
	printf("\n");


	//float *h_pool=(float *)malloc(sizeof(float)*popSize*(torCount+7));
	//cudaMemcpy(h_pool,pool,sizeof(float)*popSize*(torCount+7),cudaMemcpyDeviceToHost);

	//float *h_score=(float *) malloc(sizeof(float)*popSize);
	//cudaMemcpy(h_score,ga_score,sizeof(float)*popSize,cudaMemcpyDeviceToHost);

	//for (int i=0;i<popSize;i++)
	//{
		//printf("Ind %3d (%10.3f) [",i,h_score[i]);
			//for (int j=0;j<7;++j)
				//printf("%6.3f ",h_pool[j*popSize+i]);
			//for (int j=7;j<7+torCount;++j)
				//printf("%6.1f ",h_pool[j*popSize+i]*2.0f*57.29577951f);
			//printf("\n");
	//}


	for (int i=0;i<mapCount;i++)
		delete tex[i];
	delete vdw_map;
	delete desol;
	delete epsilon;
	delete r_eps;
	free(maps);
	free(best_score);
	free(best_var);
	//free(h_pool);
	//free(h_score);
	cudaFree(nonBond);
	cudaFree(d_rngStates);
	cudaFree(pool);
	cudaFree(ga_score);
	cudaDeviceReset();
	//printf("Reseted\n");
	return 0;
}
