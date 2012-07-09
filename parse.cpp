#include "parse.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define TYPECOUNT 25

int parse_dpf(char * line)
{
	char temp[1000];
	sscanf(line,"%s",temp);
	for (int i=0;i<5;i++)
	{
		if (strcmp(keyword[i],temp)==0)
			return i;
	}
	return -1;
}


int getType(char * atom)
{
		for (int i=0;i<strlen(atom);i++)
		{
			if (atom[i]>='a' && atom[i]<='z')
				atom[i]+=('A'-'a');
		}
	if (strcmp(atom,"E")==0) return 200;
	else if (strcmp(atom,"D")==0) return 300;
	else
	{
		for (int i=0;i<TYPECOUNT;i++)
		{
			if (strcmp(atom,atmName[i])==0)
				return i;
		}
	}
	return -1;
}

int getIDX(char * name)
{
//	printf("%s\n",name);
	char atom[1000];
	int p=strlen(name)-1;
	int x=-1,y=-1;
	while (p>=0)
	{
		if (name[p]=='.' && y>=0)
			x=p;
		if (name[p]=='.' && y<0)
			y=p;
		if (x>=0 && y>=0)
			break;
		p--;
	}
//	printf("%d %d\n",x,y);
	if (x>=y) 
	{
		printf("error\n");
		return -1;
	}
		strncpy(atom,name+x+1,y-x-1);
		atom[y-x-1]='\0';

	return getType(atom);
}

