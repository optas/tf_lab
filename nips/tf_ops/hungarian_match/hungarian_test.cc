#include <cmath>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdio.h>

using namespace std;

#define INF 100000000 //just infinity

// Helper functions for hungarian
void init_labels(int n, const float* cost, float* lx, float* ly)
{
	memset(lx, 0, sizeof(float)*n);
	memset(ly, 0, sizeof(float)*n);
        //for (int i=0; i<n; i++) lx[i] = 0;
        //for (int i=0; i<n; i++) ly[i] = 0;
	for (int x = 0; x < n; x++)
		for (int y = 0; y < n; y++)
			lx[x] = std::max(lx[x], cost[x*n+y]);
}
void update_labels(int n, const bool* S, const bool* T, float* slack, float* lx, float* ly)
{
	int x, y;
	float delta = INF; //init delta as infinity
	for (y = 0; y < n; y++) //calculate delta using slack
		if (!T[y])
			delta = std::min(delta, slack[y]);
	for (x = 0; x < n; x++) //update X labels
		if (S[x]) lx[x] -= delta;
	for (y = 0; y < n; y++) //update Y labels
		if (T[y]) ly[y] += delta;
	for (y = 0; y < n; y++) //update slack array
		if (!T[y])
			slack[y] -= delta;
}
void add_to_tree(int n, int x, int prevx, const float* lx, const float* ly, const float* cost, bool* S, int* prev, float* slack, int* slackx) 
	//x - current vertex,prevx - vertex from X before x in the alternating path,
	//so we add edges (prevx, xy[x]), (xy[x], x)
{
	S[x] = true; //add x to S
	prev[x] = prevx; //we need this when augmenting
	for (int y = 0; y < n; y++) //update slacks, because we add new vertex to S
		if (lx[x] + ly[y] - cost[x*n+y] < slack[y])
		{
			slack[y] = lx[x] + ly[y] - cost[x*n+y];
			slackx[y] = x;
		}
}
void augment(int n, int max_match, const float* cost, bool* S, bool* T, int* prev, int* xy, int* yx, float* slack, int* slackx, float* lx, float* ly) //main function of the algorithm
{
	if (max_match == n) return; //check wether matching is already perfect
	int x, y, root; //just counters and root vertex
	int q[n];
	int wr = 0, rd = 0; //q - queue for bfs, wr,rd - write and read
	//pos in queue
	memset(S, false, sizeof(bool)*n); //init set S
	memset(T, false, sizeof(bool)*n); //init set T
	//for (int i=0; i<n; i++) S[i] = false;
	//for (int i=0; i<n; i++) T[i] = false;


	for (int i=0; i<n; i++) prev[i] = -1;
	//memset(prev, -1, sizeof(prev)); //init set prev - for the alternating tree
	for (x = 0; x < n; x++) //finding root of the tree
		if (xy[x] == -1)
		{
			q[wr++] = root = x;
			prev[x] = -2;
			S[x] = true;
			break;
		}

	for (y = 0; y < n; y++) //initializing slack array
	{
		slack[y] = lx[root] + ly[y] - cost[root*n+y];
		slackx[y] = root;
	}

	//second part of augment() function
	while (true) //main cycle
	{
		cout << "in while loop.." << endl;
		while (rd < wr) //building tree with bfs cycle
		{
			x = q[rd++]; //current vertex from X part
			for (y = 0; y < n; y++) {//iterate through all edges in equality graph
				printf("## %f\n", fabs(cost[x*n+y] - lx[x] - ly[y]));
				if (fabs(cost[x*n+y] - lx[x] - ly[y]) < 1e-6 && !T[y])
				{
					if (yx[y] == -1) break; //an exposed vertex in Y found, so
					//augmenting path exists!
					T[y] = true; //else just add y to T,
					q[wr++] = yx[y]; //add vertex yx[y], which is matched
					//with y, to the queue
					add_to_tree(n, yx[y], x, lx, ly, cost, S, prev, slack, slackx); //add edges (x,y) and (y,yx[y]) to the tree
				}
			}
			if (y < n) break; //augmenting path found!
		}
		if (y < n) break; //augmenting path found!

		update_labels(n, S, T, slack, lx, ly); //augmenting path not found, so improve labeling
		wr = rd = 0; 
		for (y = 0; y < n; y++) 
			//in this cycle we add edges that were added to the equality graph as a
			//result of improving the labeling, we add edge (slackx[y], y) to the tree if
			//and only if !T[y] && slack[y] == 0, also with this edge we add another one
			//(y, yx[y]) or augment the matching, if y was exposed
			if (!T[y] && fabs(slack[y]) < 1e-6)
			{
				if (yx[y] == -1) //exposed vertex in Y found - augmenting path exists!
				{
					x = slackx[y];
					break;
				}
				else
				{
					T[y] = true; //else just add y to T,
					if (!S[yx[y]]) 
					{
						q[wr++] = yx[y]; //add vertex yx[y], which is matched with
						//y, to the queue
						add_to_tree(n, yx[y], slackx[y], lx, ly, cost, S, prev, slack, slackx); //and add edges (x,y) and (y,
						//yx[y]) to the tree
					}
				}
			}
		if (y < n) break; //augmenting path found!
	}
	cout << "funk" << endl;
	if (y < n) //we found augmenting path!
	{
		max_match++; //increment matching
		//in this cycle we inverse edges along augmenting path
		for (int cx = x, cy = y, ty; cx != -2; cx = prev[cx], cy = ty)
		{
			ty = xy[cx];
			yx[cy] = cx;
			xy[cx] = cy;
		}
		augment(n, max_match, cost, S, T, prev, xy, yx, slack, slackx, lx, ly); //recall function, go to step 1 of the algorithm
	}
}//end of augment() function





void hungarian(int n, const float* cost, float* match) {
	int max_match; //n workers and n jobs
	float lx[n], ly[n]; //labels of X and Y parts
	int xy[n]; //xy[x] - vertex that is matched with x,
	int yx[n]; //yx[y] - vertex that is matched with y
	bool S[n], T[n]; //sets S and T in algorithm
	float slack[n]; //as in the algorithm description
	int slackx[n]; //slackx[y] such a vertex, that
	// l(slackx[y]) + l(y) - w(slackx[y],y) = slack[y]
	int prev[n]; //array for memorizing alternating paths
	max_match = 0;
	for (int i=0; i<n; i++) xy[i] = -1;
	for (int i=0; i<n; i++) yx[i] = -1;
	//memset(xy, -1, sizeof(xy));
	//memset(yx, -1, sizeof(yx));
	init_labels(n, cost, lx, ly);
	augment(n, max_match, cost, S, T, prev, xy, yx, slack, slackx, lx, ly);

	float ret = 0;
	for (int x = 0; x < n; x++) //forming answer there
	{
		ret += cost[x*n+xy[x]];
		cout << x << " " << xy[x] << endl;
	}

	for (int i=0; i<n; i++) {
		match[i*n+xy[i]] = 1;
	}
}

int main() {
	int n = 3;
	float cost[n*n];
	cost[0]=2;
	cost[1]=3;
	cost[2]=3;
	cost[3]=1;
	cost[4]=10;
	cost[5]=30;
	cost[6]=3;
	cost[7]=3;
	cost[8]=2;
	float match[n*n];
	memset(match, 0, sizeof(match));
	hungarian(n, cost, match);
	for (int i=0;i<n;i++)
		for (int j=0;j<n;j++)
			cout << match[i*n+j] << endl;
}
