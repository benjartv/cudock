#ifndef __ENERGY_RULER__
#define __ENERGY_RULER__

#include "parse.h"
#include "molecule.h"
#include <vector>
#include <bitset>
#include "distdepdiel.h"
#include <math.h>

using namespace std;

#define MAX_TYPE 27
#define EINTCLAMP 10000.0

#define coeff_vdw 0.1662
#define coeff_hbond 0.1209
//#define coeff_vdw 1.0
//#define coeff_hbond 1.0
#define coeff_estat 0.1406
#define coeff_desolv 0.1322
#define coeff_tors   0.2983

class energyRuler
{
public:

	float * vdw_table;
	
	float sol_table[2000];

	float epsilon[2000],r_epsilon[2000];
	
	void print()
	{
		for (int i=0;i<1000;i++)
		{
			printf("epsilon @ %.4f angstrom:\t%f\n",(float)i*0.01,epsilon[i]);
		}

		for (int i=0;i<1000;i++)
		{
			printf("%f\n",vdw_table[0*MAX_TYPE*MAX_TYPE*1000+1*MAX_TYPE*1000+i]);
		}
	}

	~energyRuler()
	{
		free(vdw_table);
	}

	energyRuler()
	{

//		printf("inside ruler\n");

		bitset<100> log;
		log.reset();

		memset(epsilon,0,sizeof(epsilon));
		memset(r_epsilon,0,sizeof(r_epsilon));

//		printf("init\n");

		vdw_table=(float *)malloc(sizeof(float)*1000*MAX_TYPE*MAX_TYPE);
		memset(vdw_table,0,sizeof(vdw_table));
/*
		for (int i=0;i<MAX_TYPE;i++)
			for (int j=0;j<MAX_TYPE;j++)
			{
				vdw_table[i][j]=(float *)malloc(sizeof(float)*1000);
				for (int k=0;k<1000;k++)
					vdw_table[i][j][k]=0;
			}
*/
//		for (int i=0;i<types.size();i++)
//			printf("%d ",types[i]);
//		printf("\n");

//		printf("doing solv and epsilon\n");

		for (int index=0;index<1000;++index)
		{
			float r=(float)index*0.01f;
			sol_table[index]=exp(-0.03858f*r*r)*0.1322;
		}

		for (int index=0;index<1000;++index)
		{
			float r=(float)index*0.01f;
			epsilon[index]=calc_ddd_Mehler_Solmajer(r,1e-6);
			r_epsilon[index]=(coeff_estat*332.06363)/(r*epsilon[index]);
		}
		r_epsilon[0]=EINTCLAMP;

//		printf("doing vdw\n");

		for (int i=0;i<MAX_TYPE;++i)
		{
			float Ri=radius[i];
			float epsi=eps[i]*coeff_vdw;
			float R_hb_i=radius_hb[i];
			float eps_hb_i=eps_hb[i]*coeff_hbond;
			float hbond_i=hbond[i];
			for (int j=i;j<MAX_TYPE;++j)
			{

				float Rj=radius[j];
				float epsj=eps[j]*coeff_vdw;
				float R_hb_j=radius_hb[j];
				float eps_hb_j=eps_hb[j]*coeff_hbond;
				float hbond_j=hbond[j];

				float xA=12, xB=6;

				float Rij=0, epsij=0;
				if ( ((hbond_i==1)||(hbond_i==2)) && ((hbond_j>=3)&&(hbond_j<=5)))
				{
					Rij=R_hb_j;
					epsij=eps_hb_j;
					xB=10;
				}
				else if ( ((hbond_j==1)||(hbond_j==2)) && ((hbond_i>=3)&&(hbond_i<=5)))
				{
					Rij=R_hb_i;
					epsij=eps_hb_i;
					xB=10;
				}
				else
				{
					Rij=(Ri+Rj)*0.5;
					epsij=sqrt(epsi*epsj);
				}

				if (xA!=xB)
				{
					float cA=epsij*pow(Rij,xA)*(xB/(xA-xB));
					float cB=epsij*pow(Rij,xB)*(xA/(xA-xB));

					for (int index=1;index<1000;++index)
					{
						
		//				printf("%6d %6d %6d: %6d %6d\n",i,j,index,i*MAX_TYPE*1000+j*1000+index,j*MAX_TYPE*1000+i*1000+index);

						float r=(float)index*0.01f;
						float rA=pow(r,xA);
						float rB=pow(r,xB);

						float energy=(cA/rA-cB/rB);
						if (energy>EINTCLAMP) energy=EINTCLAMP;

						vdw_table[i*MAX_TYPE*1000+j*1000+index]=energy;
						vdw_table[j*MAX_TYPE*1000+i*1000+index]=energy;
					}
					vdw_table[i*MAX_TYPE*1000+j*1000+0]=EINTCLAMP;
					vdw_table[j*MAX_TYPE*1000+i*1000+0]=EINTCLAMP;
				}
			}

		}
	}
};

#endif
