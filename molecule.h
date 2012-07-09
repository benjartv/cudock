#ifndef __MOLECULE__
#define __MOLECULE__

#include <vector>
#include <bitset>
#include <iostream>
#include <string>
#include <utility>
#include <map>
#include "parse.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

#define BOND_LENGTH_TOLERANCE 0.1
#define set_minmax( a1, a2, min, max)  \
	    mindist[(a1)][(a2)] = mindist[(a2)][(a1)] = (min)-BOND_LENGTH_TOLERANCE;\
	    maxdist[(a1)][(a2)] = maxdist[(a2)][(a1)] = (max)+BOND_LENGTH_TOLERANCE

enum {C=0,N=1,O=2,H=3,XX=4,P=5,S=6};
#define NUM_ENUM_ATOMTYPES 7
static double mindist[NUM_ENUM_ATOMTYPES][NUM_ENUM_ATOMTYPES];
static double maxdist[NUM_ENUM_ATOMTYPES][NUM_ENUM_ATOMTYPES];

static bool sizeComp(bitset<1000> a, bitset<1000> b)
{
	return a.count()<b.count();
}

static bool distComp(pair<int, float> a, pair<int, float> b)
{
	return a.second<b.second;
}

class ligand
{
public:
      vector<float> coord_x;
      vector<float> coord_y;
      vector<float> coord_z;
      vector<float> elecq;
      vector<int>   type;

	  vector<int> piece;

	  vector< bitset<1000> > torsion;
	  vector< pair<int, int> > base;
	  bitset<1000> ignore;

	  short nonbonded[1000][1000];

   /*
      vector<int> rot_0;
      vector<int> rot_1;
      vector< bitset<1000> > moved;
     */ 

	  int atomCount;
	  int flexCount;
      int torCount;
      float cent_x,cent_y,cent_z;


	  ligand & operator = (const ligand & rhand)
	  {
		  coord_x=rhand.coord_x;
		  coord_y=rhand.coord_y;
		  coord_z=rhand.coord_z;
		  elecq=rhand.elecq;
		  type=rhand.type;
		  piece=rhand.piece;
		  torsion=rhand.torsion;
		  base=rhand.base;
		  ignore=rhand.ignore;

		  for (int i=0;i<1000;i++)
			  for (int j=0;j<1000;j++)
				  nonbonded[i][j]=rhand.nonbonded[i][j];

		  atomCount=rhand.atomCount;
		  flexCount=rhand.flexCount;
		  torCount=rhand.torCount;

		  cent_x=rhand.cent_x;
		  cent_y=rhand.cent_y;
		  cent_z=rhand.cent_z;
		  return *this;
	  }


       void print_atom()
       {
            for (int i=0;i<atomCount;i++)
            {
                printf("%d [%.4f %.4f %.4f] (%.4f) %s\n",i,coord_x[i],coord_y[i],coord_z[i],elecq[i],atmName[type[i]]);
            }
       }

	   inline void getCoord(int idx, float *x, float *y, float *z)
	   {
		   *x=coord_x[idx];
		   *y=coord_y[idx];
		   *z=coord_z[idx];
	   }

       void print_tor()
       {/*
            map < pair<int, int>, bitset<1000> >::iterator it;
            for (it= t_torsion.begin();it!=t_torsion.end();it++)
            {
                printf("%2d-%2d\t",it->first.first,it->first.second);
                for (int i=0;i<atomCount;i++)
                {
                    if (it->second.test(i))
                       printf("%2d,",i);
                }
                printf("\n");
            }
		*/
		if (torCount!=torsion.size() || torCount!=base.size())
		{
			printf("Error! Torsion vector size not match!\n");
			return;
		}
		for (int i=0;i<torCount;i++)
		{
			printf("%2d-%2d\t",base[i].first,base[i].second);
			for (int j=0;j<atomCount;j++)
			{
				if (torsion[i].test(j))
					printf("%2d,",j);
			}
			printf("\n");
		}
       }
       
       ligand() 
	   {
		   memset(nonbonded,0,sizeof(nonbonded));
	   }

	   ligand(char * ligName, char * flexName)
	   { 
		   memset(nonbonded,0,sizeof(nonbonded));
		   ligand lig(ligName);
		   ligand flex(flexName);

		   flexCount=flex.atomCount;
		   atomCount=lig.atomCount+flex.atomCount;
		   torCount=lig.torCount+flex.torCount;
		   cent_x=lig.cent_x;
		   cent_y=lig.cent_y;
		   cent_z=lig.cent_z;

		   coord_x= lig.coord_x;
		   coord_x.insert(coord_x.end(), flex.coord_x.begin(), flex.coord_x.end());
		   coord_y= lig.coord_y;
		   coord_y.insert(coord_y.end(), flex.coord_y.begin(), flex.coord_y.end());
		   coord_z= lig.coord_z;
		   coord_z.insert(coord_z.end(), flex.coord_z.begin(), flex.coord_z.end());

		   for (int i=lig.atomCount;i<(lig.atomCount+flex.atomCount);++i)
		   {
			   coord_x[i]+=flex.cent_x;
			   coord_y[i]+=flex.cent_y;
			   coord_z[i]+=flex.cent_z;
		   }

		   elecq= lig.elecq;
		   elecq.insert(elecq.end(), flex.elecq.begin(), flex.elecq.end());
		   type= lig.type;
		   type.insert(type.end(), flex.type.begin(), flex.type.end());

		   ignore=( lig.ignore | flex.ignore << lig.atomCount );

		   vector< pair<int, int> > swap;
		   vector< bitset<1000> > unsorted;
		   unsorted=lig.torsion;
		   unsorted.insert(unsorted.end(),flex.torsion.begin(),flex.torsion.end());

		   swap=lig.base;
		   swap.insert(swap.end(),flex.base.begin(), flex.base.end());
		   for (int i=lig.base.size();i<(lig.base.size()+flex.base.size());++i)
		   {
			   swap[i].first+=lig.atomCount;
			   swap[i].second+=lig.atomCount;
			   unsorted[i]=(unsorted[i] << lig.atomCount);
		   }
		   torsion=unsorted;
		   stable_sort(torsion.begin(),torsion.end(),sizeComp);
		   base.resize(torsion.size());
		   for (int i=0;i<torsion.size();++i)
		   {
			   for (int j=0;j<torsion.size();++j)
			   {
				   if (torsion[i]==unsorted[j])
				   {
					   base[i]=swap[j];
					   break;
				   }
			   }
		   }

		   for (int i=0;i<atomCount;i++)
			   for (int j=0;j<atomCount;j++)
			   {
				   nonbonded[i][j]=1;
			   }
		   for (int i=0;i<lig.atomCount;i++)
			   for (int j=0;j<lig.atomCount;j++)
			   {
				   nonbonded[i][j]=lig.nonbonded[i][j];
			   }
		   for (int i=0;i<flex.atomCount;i++)
			   for (int j=0;j<flex.atomCount;j++)
			   {
				   nonbonded[i+lig.atomCount][j+lig.atomCount]=flex.nonbonded[i][j];
			   }
	   } 

       ligand(char * fileName)
       {
		   memset(nonbonded,0,sizeof(nonbonded));
		   ignore.reset();
		   flexCount=0;
           vector< pair<int, int> > branch;
	       map< pair<int, int>, bitset<1000> > t_torsion;
		   map< pair<int, int>, int> t_serial;
           cent_x=0;
           cent_y=0;
           cent_z=0;
           atomCount=0;
           torCount=0;
		   
		   int piece_index=0;


           bool new_res=false;
		   int natom_in_res=0;

           FILE * file=fopen(fileName,"r");
           char line[1000];
           while (fgets(line,1000,file)!=NULL)
           {
                if (strncmp(line,"BRANCH",6)==0)
    			{
    				int a,b;
    				sscanf(line,"%*s %d %d",&a,&b);
    				pair<int, int> bond(a-1,b-1);
    				bitset<1000> zero(0u);
    				branch.push_back(bond);
    				
    				t_torsion[bond]=zero;
					t_serial[bond]=torCount;

    				torCount++;
					piece_index++;
    			}
				else if (strncmp(line,"ENDBRANCH",9)==0)
    			{
    				int a,b;
    				sscanf(line,"%*s %d %d",&a,&b);
    				pair<int, int> bond(a-1,b-1);
    				if (branch.back()==bond)
    				{
    					branch.pop_back();
    				}
    			}
				else if (strncmp(line,"ATOM",4)==0 || strncmp(line,"HETATM",6)==0)
    			{
					if (new_res)
					{
						if (natom_in_res<2)
							ignore.set(atomCount);
						natom_in_res++;
					}
					else
					{
						natom_in_res=0;
					}
    				char temp[10];
    				float x,y,z,q;
    				//30-37 x
    				strncpy(temp,line+30,8);
    				temp[8]='\0';
    				sscanf(temp,"%f",&x);
    				coord_x.push_back(x);
    				//38-45 y
    				strncpy(temp,line+38,8);
    				temp[8]='\0';
    				sscanf(temp,"%f",&y);
    				coord_y.push_back(y);
    				//46-53 z
    				strncpy(temp,line+46,8);
    				temp[8]='\0';
    				sscanf(temp,"%f",&z);
    				coord_z.push_back(z);
    				//70-75 q
    				strncpy(temp,line+70,6);
    				temp[6]='\0';
    				sscanf(temp,"%f",&q);
    				elecq.push_back(q);
    
    				cent_x+=x;
    				cent_y+=y;
    				cent_z+=z;
    
    				strncpy(temp,line+77,2);
    				if (temp[1]==' ') temp[1]='\0';
    				else temp[2]='\0';
    				type.push_back(getType(temp));
    				
    				for (int i=0;i<branch.size();i++)
    				{
                        if (atomCount!=branch[i].first && atomCount!=branch[i].second)
                        {
                           t_torsion[branch[i]].set(atomCount);
                        }
                    }

					piece.push_back(piece_index);

                    atomCount++;
    			}
				else if (strncmp(line,"BEGIN_RES",9)==0)
				{
					new_res=true;
					natom_in_res=0;
					piece_index++;
				}
				else if (strncmp(line,"END_RES",7)==0)
				{
					new_res=false;
				}
    		}
    		fclose(file);

			cent_x=cent_x/((float)atomCount);
			cent_y=cent_y/((float)atomCount);
			cent_z=cent_z/((float)atomCount);
			for (int i=0;i<atomCount;i++)
			{
				coord_x[i]-=cent_x;
				coord_y[i]-=cent_y;
				coord_z[i]-=cent_z;
			}

            map < pair<int, int>, bitset<1000> >::iterator it;
			torsion.resize(torCount);
			base.resize(torCount);
            for (it= t_torsion.begin();it!=t_torsion.end();it++)
            {
				pair <int, int> bond=it->first;
                torsion[t_serial[bond]]=it->second;
			}
			stable_sort(torsion.begin(),torsion.end(),sizeComp);

			for (int i=0;i<torsion.size();i++)
			{
				for (it=t_torsion.begin();it!=t_torsion.end();it++)
				{
					if (torsion[i]==it->second)
					{
						base[i]=it->first;
					}
				}
			}

			checkNonBond();
	   }

	   void checkNonBond()
	   {
//		   printf("in checkNonBond\n");
		   short bond[atomCount][6];
		   for (int i=0;i<atomCount;++i)
		   {
			   for (int j=0;j<5;++j)
			   {
				   bond[i][j]=-1;
			   }
			   bond[i][5]=0;
		   }

		   set_min_max();
/*
		   for (int i=0;i<7;i++)
			   for (int j=0;j<7;++j)
			   {
				   printf("min[%d][%d]=%f\n",i,j,mindist[i][j]);
			   }
*/
		   vector< pair<int, float> > distance;

		   for (int i=0;i<atomCount;++i)
		   {
			   distance.clear();
			   for (int j=i+1;j<atomCount;++j)
			   {
				   float dx=coord_x[i]-coord_x[j];
				   float dy=coord_y[i]-coord_y[j];
				   float dz=coord_z[i]-coord_z[j];

				   pair<int, float> temp=make_pair(j,dx*dx+dy*dy+dz*dz);
				   distance.push_back(temp);
			   }
			   sort(distance.begin(),distance.end(),distComp);
			   for (int p=0;p<distance.size();++p)
			   {
				   int j=distance[p].first;
				   float dist=distance[p].second;

				   if (dist >= mindist[ bondType[type[i]] ][ bondType[type[j]] ] &&
					   dist <= maxdist[ bondType[type[i]] ][ bondType[type[j]] ] )
				   {
					   if (bond[i][5]>=5 || bond[j][5]>=5)
					   {
						   continue;
					   }
					   bond[i][ bond[i][5] ]=j;
					   bond[j][ bond[j][5] ]=i;
					   bond[i][5]++;
					   bond[j][5]++;
				   }
			   }
		   }
		   /*
		   for (int i=0;i<atomCount;++i)
		   {
			   printf("bond[%3d]\t%d\n",i,bond[i][5]);
		   }

		   printf("done bond\n");
*/
		   for (int i=0;i<atomCount;++i)
		   {
			   for (int j=0;j<atomCount;++j)
			   {
				   nonbonded[i][j]=1;
			   }
			   nonbonded[i][i]=0;
		   }
		   for (int i=0;i<atomCount;++i)
		   {
			   for (int j=0;j<bond[i][5];++j)
			   {
				   nonbonded[i][ bond[i][j] ]=0;
				   nonbonded[ bond[i][j] ][i]=0;
			   }
		   }

//		   printf("done direct\n");
/*
		   for (int i=0;i<atomCount;++i)
		   {
			   printf("bond[%2d] ",i+1);
			   for (int j=0;j<bond[i][5];++j)
				   printf("%d ",bond[i][j]);
			   printf("%d\n",bond[i][5]);
		   }
*/
		   for (int i=0;i<atomCount;++i)
			   for (int j=0;j<bond[i][5];++j)
				   for (int k=0;k<bond[ bond[i][j] ][5]; k++)
				   {
					   nonbonded[ bond[ bond[i][j] ][k] ][i]=0;
					   nonbonded[i][ bond[ bond[i][j] ][k] ]=0;
					   
					   for (int l=0;l<bond[ bond[ bond[i][j]  ][k]  ][5]; l++)
					   {
						   nonbonded[i][ bond[ bond[ bond[i][j]][k]][l]  ] =0;
						   nonbonded[ bond[ bond[ bond[i][j]][k]][l]  ][i] =0;
					   }
					   
				   }


//		   printf("done wtf!\n");
		   for (int i=0;i<atomCount;++i)
		   {
//			   printf("piece[%3d] = %d\n",i+1,piece[i]);
			   for (int j=i+1;j<atomCount;++j)
			   {
				   if (piece[i]==piece[j])
				   {
					   nonbonded[i][j]=0;
					   nonbonded[j][i]=0;
				   }
			   }
		   }

		   for (int i=0;i<torCount;++i)
		   {
			   int a11=base[i].first;
			   int a21=base[i].second;

			   for (int j=0;j<torCount;++j)
			   {
				   int a12=base[j].first;
				   int a22=base[j].second;

				   if (piece[a11]==piece[a12]) 
					   nonbonded[a22][a21]=nonbonded[a21][a22]=0;
				   if (piece[a11]==piece[a22]) 
					   nonbonded[a12][a21]=nonbonded[a21][a12]=0;
				   if (piece[a21]==piece[a12]) 
					   nonbonded[a22][a11]=nonbonded[a11][a22]=0;
				   if (piece[a21]==piece[a22]) 
					   nonbonded[a12][a11]=nonbonded[a11][a12]=0;
			   }

			   for (int k=0;k<atomCount;k++)
			   {
				   int p=piece[k];
				   if (piece[a11]==p)
					   nonbonded[k][a21]=nonbonded[a21][k]=0;
				   if (piece[a21]==p)
					   nonbonded[k][a11]=nonbonded[a11][k]=0;
			   }
		   }
	   }


	   void set_min_max()
	   { 
		   //taken from autodock 4.2 source code
    set_minmax(C, C, 1.20, 1.545); // mindist[C][C] = 1.20, p. 3510 ; maxdist[C][C] = 1.545, p. 3511
    set_minmax(C, N, 1.1, 1.479); // mindist[C][N] = 1.1, p. 3510 ; maxdist[C][N] = 1.479, p. 3511
    set_minmax(C, O, 1.15, 1.47); // mindist[C][O] = 1.15, p. 3510 ; maxdist[C][O] = 1.47, p. 3512
    set_minmax(C, H, 1.022, 1.12);  // p. 3518, p. 3517
    set_minmax(C, XX, 0.9, 1.545); // mindist[C][XX] = 0.9, AutoDock 3 defaults ; maxdist[C][XX] = 1.545, p. 3511
    set_minmax(C, P, 1.85, 1.89); // mindist[C][P] = 1.85, p. 3510 ; maxdist[C][P] = 1.89, p. 3510
    set_minmax(C, S, 1.55, 1.835); // mindist[C][S] = 1.55, p. 3510 ; maxdist[C][S] = 1.835, p. 3512
    set_minmax(N, N, 1.0974, 1.128); // mindist[N][N] = 1.0974, p. 3513 ; maxdist[N][N] = 1.128, p. 3515
    set_minmax(N, O, 1.0619, 1.25); // mindist[N][O] = 1.0975, p. 3515 ; maxdist[N][O] = 1.128, p. 3515
    set_minmax(N, H, 1.004, 1.041); // mindist[N][H] = 1.004, p. 3516 ; maxdist[N][H] = 1.041, p. 3515
    set_minmax(N, XX, 0.9, 1.041); // mindist[N][XX] = 0.9, AutoDock 3 defaults ; maxdist[N][XX] = 1.041, p. 3515
    set_minmax(N, P, 1.4910, 1.4910); // mindist[N][P] = 1.4910, p. 3515 ; maxdist[N][P] = 1.4910, p. 3515
    set_minmax(N, S, 1.58, 1.672); // mindist[N][S] = 1.58, 1czm.pdb sulfonamide ; maxdist[N][S] = 1.672, J. Chem. SOC., Dalton Trans., 1996, Pages 4063-4069 
    set_minmax(O, O, 1.208, 1.51); // p.3513, p.3515
    set_minmax(O, H, 0.955, 1.0289); // mindist[O][H] = 0.955, p. 3515 ; maxdist[O][H] = 1.0289, p. 3515
    set_minmax(O, XX, 0.955, 2.1); // AutoDock 3 defaults
    set_minmax(O, P, 1.36, 1.67); // mindist[O][P] = 1.36, p. 3516 ; maxdist[O][P] = 1.67, p. 3517
    set_minmax(O, S, 1.41, 1.47); // p. 3517, p. 3515
    set_minmax(H, H, 100.,-100.); // impossible values to prevent such bonds from forming.
    set_minmax(H, XX, 0.9, 1.5); // AutoDock 4 defaults
    set_minmax(H, P, 1.40, 1.44); // mindist[H][P] = 1.40, p. 3515 ; maxdist[H][P] = 1.44, p. 3515
    set_minmax(H, S, 1.325, 1.3455); // mindist[H][S] = 1.325, p. 3518 ; maxdist[H][S] = 1.3455, p. 3516
    set_minmax(XX, XX, 0.9, 2.1); // AutoDock 3 defaults
    set_minmax(XX, P, 0.9, 2.1); // AutoDock 3 defaults
    set_minmax(XX, S, 1.325, 2.1); // mindist[XX][S] = 1.325, p. 3518 ; maxdist[XX][S] = 2.1, AutoDock 3 defaults
    set_minmax(P, P, 2.18, 2.23); // mindist[P][P] = 2.18, p. 3513 ; maxdist[P][P] = 2.23, p. 3513
    set_minmax(P, S, 1.83, 1.88); // mindist[P][S] = 1.83, p. 3516 ; maxdist[P][S] = 1.88, p. 3515
    set_minmax(S, S, 2.03, 2.05); // mindist[S][S] = 2.03, p. 3515 ; maxdist[S][S] = 2.05, p. 3515
	for (int i=0;i<NUM_ENUM_ATOMTYPES;i++)
		for (int j=0;j<NUM_ENUM_ATOMTYPES;j++)
		{
			mindist[i][j]*=mindist[i][j];
			maxdist[i][j]*=maxdist[i][j];
		}
	   }
};


#endif
