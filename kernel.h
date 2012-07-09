#ifndef __KERNEL__
#define __KERNEL__

// psedo text array access
//
#define comp_min(_x,_y) (((_x)>(_y))?(_y):(_x))
#define comp_max(_x,_y) (((_x)>(_y))?(_x):(_y))

#define arrayTexFetch(_index, _tu, _tv, _tw, _return) \
{\
	switch(_index)\
	{\
		case 0 :_return=tex3D(tex_H ,(_tu),(_tv),(_tw));break;\
		case 1 :_return=tex3D(tex_HD,(_tu),(_tv),(_tw));break;\
		case 2 :_return=tex3D(tex_HS,(_tu),(_tv),(_tw));break;\
		case 3 :_return=tex3D(tex_C ,(_tu),(_tv),(_tw));break;\
		case 4 :_return=tex3D(tex_A ,(_tu),(_tv),(_tw));break;\
		case 5 :_return=tex3D(tex_N ,(_tu),(_tv),(_tw));break;\
		case 6 :_return=tex3D(tex_NA,(_tu),(_tv),(_tw));break;\
		case 7 :_return=tex3D(tex_NS,(_tu),(_tv),(_tw));break;\
		case 8 :_return=tex3D(tex_OA,(_tu),(_tv),(_tw));break;\
		case 9 :_return=tex3D(tex_OS,(_tu),(_tv),(_tw));break;\
		case 10:_return=tex3D(tex_F ,(_tu),(_tv),(_tw));break;\
		case 11:_return=tex3D(tex_Mg,(_tu),(_tv),(_tw));break;\
		case 12:_return=tex3D(tex_P ,(_tu),(_tv),(_tw));break;\
		case 13:_return=tex3D(tex_SA,(_tu),(_tv),(_tw));break;\
		case 14:_return=tex3D(tex_S ,(_tu),(_tv),(_tw));break;\
		case 15:_return=tex3D(tex_Cl,(_tu),(_tv),(_tw));break;\
		case 16:_return=tex3D(tex_Ca,(_tu),(_tv),(_tw));break;\
		case 17:_return=tex3D(tex_Mn,(_tu),(_tv),(_tw));break;\
		case 18:_return=tex3D(tex_Fe,(_tu),(_tv),(_tw));break;\
		case 19:_return=tex3D(tex_Zn,(_tu),(_tv),(_tw));break;\
		case 20:_return=tex3D(tex_Br,(_tu),(_tv),(_tw));break;\
		case 21:_return=tex3D(tex_I ,(_tu),(_tv),(_tw));break;\
		case 22:_return=tex3D(tex_Z ,(_tu),(_tv),(_tw));break;\
		case 23:_return=tex3D(tex_G ,(_tu),(_tv),(_tw));break;\
		case 24:_return=tex3D(tex_GA,(_tu),(_tv),(_tw));break;\
		case 25:_return=tex3D(tex_J ,(_tu),(_tv),(_tw));break;\
		case 26:_return=tex3D(tex_Q ,(_tu),(_tv),(_tw));break;\
		case 200:_return=tex3D(tex_e,(_tu),(_tv),(_tw));break;\
		case 300:_return=tex3D(tex_d,(_tu),(_tv),(_tw));break;\
	}\
}

char texName[][10]={
	"tex_H",
	"tex_HD",
	"tex_HS",
	"tex_C",
	"tex_A",
	"tex_N",
	"tex_NA",
	"tex_NS",
	"tex_OA",
	"tex_OS",
	"tex_F",
	"tex_Mg",
	"tex_P",
	"tex_SA",
	"tex_S",
	"tex_Cl",
	"tex_Ca",
	"tex_Mn",
	"tex_Fe",
	"tex_Zn",
	"tex_Br",
	"tex_I",
	"tex_Z",
	"tex_G",
	"tex_GA",
	"tex_J",
	"tex_Q",
};

texture <float,3,cudaReadModeElementType> tex_H;	//0
texture <float,3,cudaReadModeElementType> tex_HD;	//1
texture <float,3,cudaReadModeElementType> tex_HS;	//2
texture <float,3,cudaReadModeElementType> tex_C;	//3
texture <float,3,cudaReadModeElementType> tex_A;	//4
texture <float,3,cudaReadModeElementType> tex_N;	//5
texture <float,3,cudaReadModeElementType> tex_NA;	//6
texture <float,3,cudaReadModeElementType> tex_NS;	//7
texture <float,3,cudaReadModeElementType> tex_OA;	//8
texture <float,3,cudaReadModeElementType> tex_OS;	//9
texture <float,3,cudaReadModeElementType> tex_F;	//10
texture <float,3,cudaReadModeElementType> tex_Mg;	//11
texture <float,3,cudaReadModeElementType> tex_P;	//12
texture <float,3,cudaReadModeElementType> tex_SA;	//13
texture <float,3,cudaReadModeElementType> tex_S;	//14
texture <float,3,cudaReadModeElementType> tex_Cl;	//15
texture <float,3,cudaReadModeElementType> tex_Ca;	//16
texture <float,3,cudaReadModeElementType> tex_Mn;	//17
texture <float,3,cudaReadModeElementType> tex_Fe;	//18
texture <float,3,cudaReadModeElementType> tex_Zn;	//19
texture <float,3,cudaReadModeElementType> tex_Br;	//20
texture <float,3,cudaReadModeElementType> tex_I;	//21
texture <float,3,cudaReadModeElementType> tex_Z;	//22
texture <float,3,cudaReadModeElementType> tex_G;	//23
texture <float,3,cudaReadModeElementType> tex_GA;	//24
texture <float,3,cudaReadModeElementType> tex_J;	//25
texture <float,3,cudaReadModeElementType> tex_Q;	//26

texture <float,3 ,cudaReadModeElementType> tex_e;	//200
texture <float,3 ,cudaReadModeElementType> tex_d;	//300

texture <float,3, cudaReadModeElementType> vdw_tables;
texture <float,1, cudaReadModeElementType> sol_ruler;
texture <float,1, cudaReadModeElementType> eps_ruler;
texture <float,1, cudaReadModeElementType> r_eps_ruler;

static const float PI2=6.28318531f;
static const float PI=3.14159265f;
static const float HALFPI=1.57079633f;

extern const int IDX_E=200;
extern const int IDX_D=300;
extern const int IDX_VDW=500;
extern const int IDX_SOL=501;
extern const int IDX_EPS=502;
extern const int IDX_R_EPS=503;

#endif
