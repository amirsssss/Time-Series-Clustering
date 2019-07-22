#include <iostream>
#include <string>
#include <array>
#include <fstream>
#include <vector>
#include <iomanip>
#include <math.h>
#include <armadillo>

using namespace std;
using namespace arma;

double dis(double a,double b){
    return abs(a-b);
}

double getDistance(rowvec a, rowvec b){
    mat matrix;
    int m=a.size();
    int n=b.size();

    matrix.ones(m,n);
    matrix*=10000;
    matrix(0,0)=dis(a[0],b[0]);

    for(int i=1;i<n;i++){
        matrix(0,i)=dis(a[0],b[i])+matrix(0,i-1);
    }

    for(int i=1;i<m;i++){
        matrix(i,0)=dis(a[i],b[0])+matrix(i-1,0);
    }

    for(int i=1;i<m;i++){
        for(int j=1;j<n;j++){
            double cost=dis(a[i],b[j]);
            matrix(i,j)=cost+min(matrix(i-1,j-1),matrix(i-1,j)+matrix(i,j-1));
        }
    }
    double res=matrix(m-1,n-1)/(m+n);
    return matrix(m-1,n-1)/(m+n);
}

mat vec2mat(vector<vector<double> >&vec){
    int rows = vec.size();
    int cols = vec[0].size();
    mat A(rows, cols);
    for(int i = 0; i<rows; i++){
        for(int j=0; j<cols; j++){
            A(i, j) = vec[i][j];
        }
    }
    return A;
}

struct Kmeans_Res{
    vector <rowvec> cntr;
    vector<vector<int>> clstr;
    double dstrtion;
};


double which_is_nearest(vector<rowvec>& centroids, rowvec pt){
    double minDistance = 0;
    int minLabel = 0;
    for(int i=0; i<centroids.size(); i++){
        double tempDistance = getDistance(centroids[i], pt);
        if(i == 0|| tempDistance < minDistance){
            minDistance = tempDistance;
            minLabel = i;
        }
    }
    return minLabel;
}

double getDistortionFunction(mat data, vector<vector<int> >& cluster, vector<rowvec>& centroids){

    int nSamples = data.n_rows;
    int nDim = data.n_cols;
    double SumDistortion = 0.0;
    for(int i = 0; i < cluster.size(); i++){
        for(int j = 0; j < cluster[i].size(); j++){
            double temp = getDistance(data.row(cluster[i][j]), centroids[i]);
            SumDistortion += temp;
        }
    }
    return SumDistortion;
}



Kmeans_Res kmeans(vector<vector<double> >&input_vectors, int K){

    mat data = vec2mat(input_vectors);
    int nSamples = data.n_rows;
    int nDim = data.n_cols;
    //randomly select k samples to initialize k centroid
    vector<rowvec> centroids;
    for(int k = 0; k < K; k ++){
        int rand_int = rand() % nSamples;
        centroids.push_back(data.row(rand_int));
    }
    //iteratively find better centroids
    vector<vector<int> > cluster;
    for(int k = 0; k < K; k ++){
        vector<int > vectemp;
        cluster.push_back(vectemp);
    }
    int counter = 0;
    while(1){
        for(int k = 0; k < K; k ++){
            cluster[k].clear();
        }
        bool converge = true;
        //for every sample, find which cluster it belongs to, 
        //by comparing the distance between it and each clusters' centroid.
        for(int m = 0; m < nSamples; m++){
            int which = which_is_nearest(centroids, data.row(m));
            cluster[which].push_back(m);
        }
        
        //for every cluster, re-calculate its centroid.
        for(int k = 0; k < K; k++){
            int cluster_size = cluster[k].size();
            rowvec vectemp = zeros<rowvec>(nDim);
            for(int i = 0; i < cluster_size; i++){
                vectemp = vectemp + data.row(cluster[k][i]) / (double)cluster_size;
            }
            if(getDistance(centroids[k], vectemp) >= 1e-9) converge = false;
            centroids[k] = vectemp;
        }
        if(converge) break;
        ++counter;
    }
    
    double distortion = getDistortionFunction(data, cluster, centroids);
    return {centroids,cluster,distortion};
    // return cluster;
}


double gap(vector<vector<double> >&input_vectors,int N,int k) {
        random_device myRDevice;
        unsigned seed=myRDevice();
        default_random_engine myREngine(seed);

        mat data = vec2mat(input_vectors);
        int a = data.n_rows;
        int b = data.n_cols;

        vector<vector<double>> rv;

        for(int j=0;j<a;j++){
            vector<double> t;
            for(int i=0;i<b;i++){
                uniform_real_distribution<double> UniDist(0.0,1.0);
                double q=UniDist(myREngine);
                t.push_back(q);
            }
            rv.push_back(t);
        }
        cout<<"Before Distortion "<<endl;
        double q=kmeans(input_vectors,k).dstrtion;
        cout<<"Distortion: "<<q<<endl;

        double wk=log(q);

        double SumE=0;

        for(int i=0;i<N;i++){
            cout<<"-- in for --"<<endl;
            double l=log(kmeans(rv,k).dstrtion);
            SumE+=l;
        }

        double E=SumE/N;
        double res=E-wk;

        cout<<"E "<<E<<endl;
        cout<<"RES "<<res<<endl;

        return res;
    }


int main(){
    vector<vector<double>> v(154106, vector<double>(57, 1));

    ifstream inputFile;
    // inputFile.open("All-transactions.txt");
    inputFile.open("All-transactions-Filtered.txt");
    

    if (!inputFile) {
        cout << "Error finding input File" << endl;
        // system("pause");
        exit(-1);
    }


    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[0].size(); j++) {
            inputFile >> v[i][j];
        }
    }

    cout << "+++++++++++++" << endl;
    
    cout<<v.size()<<endl;
    cout<<v[0].size()<<endl;


    Kmeans_Res k=kmeans(v,6);

    vector <rowvec> ab=k.cntr;
    for(int i=0;i<ab.size();i++){
        for(int j=0;j<ab[i].size();j++){
            cout<<ab[i][j]<<" ";
            }
        cout<<endl;
    }
    cout<<"===+"<<endl;
    vector<vector<int>> cl=k.clstr;

    ofstream myfile("Output-6-cluster.txt");

    if (myfile.is_open()) {
        for(int i=0;i<cl.size();i++){
            for(int j=0;j<cl[i].size();j++){
                myfile<<cl[i][j]<<" ";
                }
            myfile<<endl<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"<<endl;
            cout<<i<<endl;
        }

        } else {
            cout << "Uable to open file" << endl;
    }


    for(int i=8;i<40;i++){
        // cout<<kmeans(v,i).dstrtion<<endl;
        cout<<"Number of Clusters "<<i<<endl;
        cout<<gap(v,2,i)<<endl<<endl;
    }
    return 0;
}
