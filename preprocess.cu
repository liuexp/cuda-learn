#include<vector>
#include<cstdio>
#include "common.h"

int maxV=0,lines=0;
int m;
std::vector <std::pair<int,int> > invData;
std::vector <std::pair<int,int> > tmpV;
std::vector <std::pair<int, std::pair<int, float> > > outData;
const int N = 61578414;
//const int maxDegree = 61578414;
int maxNNZPerTurn = 0;
int *outDegree;
bool generateOutDegree = true;
bool generateCSR = true;

int *row, *col;
int *csr;

int main(){
	//readConv(probability);
	time_t time0, time1;
	double diff;
	char s[1024];
	FILE *fp = fopen(inFile,"r");
	int originalMaxV = 0;
	outDegree = (int*) malloc(sizeof(int) * (N+1));
	memset(outDegree, 0, sizeof(outDegree[0])*(N+1));
	int curline = 0;
	time(&time0);
	srand(time(NULL));
	while(fgets(s, 1024, fp) != NULL ){
		FIXLINE(s);
		char del[] = "\t ";
		if(s[0]=='#' || s[0] == '%') continue;
		//double tmp = rand()/(double)RAND_MAX;
		//if(tmp>prob)continue;
		char *t;
		int a,b;
		t=strtok(s,del);
		a=atoi(t);
		t=strtok(NULL,del);
		b=atoi(t);
		originalMaxV = max(originalMaxV, max(a,b));
		invData.push_back(std::make_pair(b,a));
		curline++;
		outDegree[a]++;
	}
	fclose(fp);
	time(&time1);
	diff = difftime(time1, time0);
	printf("here %d lines reading takes %.3f\n",curline,diff);
	time(&time0);
	m = curline;
	maxNNZPerTurn = min(GPUMEM , m);
	std::string tmp(mtxBinFile);
	std::string meta = tmp + ".meta";
	fp = fopen(meta.c_str(), "w");
	fprintf(fp, "%d %d %d", n, m, (m + maxNNZPerTurn)/maxNNZPerTurn);
	fclose(fp);
	if(generateOutDegree){
		std::string tmp1 = tmp + ".outdeg";
		fp = fopen(tmp1.c_str(), "wb");
		fwrite(outDegree, sizeof(int), originalMaxV +1, fp);
		fclose(fp);
	}
	free(outDegree);
	time(&time1);
	diff = difftime(time1, time0);
	printf("dump outDegree takes %.3fs.\n", diff);
	time(&time0);
	sort(invData.begin(), invData.end());
	row = (int *)malloc(m* sizeof(int));
	col = (int *)malloc(m * sizeof(int));
	for(int i=0;i<m;i++){
		int a = invData[i].second, b = invData[i].first;
		maxV = max(maxV, max(a,b));
		row[i] = b;
		col[i] = a;
	}
	invData.swap(tmpV);
	if(generateCSR){
		int curV = 0;
		int i=0;
		csr = (int*) malloc((n+1) * sizeof(int));
		for(;i<m;curV++){
			csr[curV] = i;
			for(;i<m&&row[i]<=curV;i++);
		}
		csr[curV] = i;
		n = curV;
	}
	time(&time1);
	diff = difftime(time1, time0);
	printf("sorting, converting takes %.3fs.\n", diff);
	time(&time0);
	if(generateCSR){
		std::string tmp1 = tmp + ".csr";
		fp = fopen(tmp1.c_str(), "wb");
		fwrite(csr, sizeof(int), n + 1, fp);
		fclose(fp);
		free(csr);
	}
	printf("csr dump done.\n");
	for(unsigned int i=0;i*maxNNZPerTurn<m;i++){
		std::stringstream basefile;
		basefile<<tmp<<"."<<i;

		std::string tmp1 = basefile.str() + ".col";
		fp = fopen(tmp1.c_str(), "wb");
		unsigned int cooOffset = i*maxNNZPerTurn;
		unsigned int cnt = min(maxNNZPerTurn, m - cooOffset);
		printf("dumping %u,%u to %s\n", i, cooOffset, tmp1.c_str());
		fwrite(&col[cooOffset], sizeof(int), cnt, fp);
		fclose(fp);

		// Calculate nCurTurn
		int nCurTurn = 0;
		int lastRow = -1;
		for(int i=cooOffset;i<cooOffset + cnt; i++){
			if(row[i] == lastRow)continue;
			lastRow = row[i];
			nCurTurn++;
		}
		 tmp1 = basefile.str() + ".meta";
		fp = fopen(tmp1.c_str(), "wb");
		fprintf(fp, "%d %d %d", cnt, nCurTurn, cooOffset);
		fclose(fp);
		//FIXME: dump row if necessary
	}
	printf("all done.\n");
	free(row);
	free(col);
	return 0;
}
