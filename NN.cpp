#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <queue>
using namespace std;

/*
Code up the Nearest Neighbord Classifier : sensetive to irrelevant information
Use Nearest Neighbor inside a "wrapper" which do various searches
1)Foward Selection
2)Backward Elimination




Overview of program:
* Leave out an instance(a row in our data set)- leftOut
* Use the rest of the data as training data for Nearest Neighbor Classifier- trainingset[]
* Test newly made classifier that was trained on rest of data set to classify leftOut- classifier(leftout)
* See if accuractely classified leftOut, if correct keep track of counter, else move on and repeat
* After finished looping on entire data set correct/n = accuracy example (3correct classifications/5 total data set) = 0.6 or 60% accuracy  


Originally we would want to test our classifier on all possible subsets 
ex) Given {f1 f2 f3 f4}
We would test 
{f1} {f2} {f3} {f4},
{f1,f2}
{f1,f3}
{f1,f4}
{f2,f3}
{f2,f4} ... and so on testing all the possible sub sets.

THIS is INFEASIBLE instead we will use two searching algorithims
fowards selection
backward elemination


-----------Three Things Necessary!

1) Nearest Neighbor classifier
-Keep training instances in memory(put them in an array)
-When new data point (leftOut) is given,compute distance to the training points
-Return class label of nearest training point

2)Leave One-out Validator(function?) 
-uses training data and classifier
-leaveOneOut(leftOut, classifier)
-given a subset of features as input
-returns an accuracy score as output

3)Search algorithim



Given dataset will have 2 classes
First Column    Rest of the Columns features    F2                  F3  F4
2.00..e+00      1.2340000e+010                  6.0668000e+000      ...
1.00..e+00      6.0668000e+000                  6.0668000e+000      ...
2.00..e+00      2.3400000e+010                  6.0668000e+000      ...
1.00..e+00      4.5645400e+010                  6.0668000e+000      ...

*/

//This class keeps track of a node in foward and backward elim and determines its accuracy and features included
class Node
{
private:
public:
    double accurate = 0;
    vector<int> featuresIncluded;
    //comparator

    //The index position will be stored
    void printFeatures()
    {
        cout << "Feature set: {";
        for (int i = 0; i < featuresIncluded.size(); i++)
        {
            cout << featuresIncluded.at(i) << ",";
        }
        cout << "} was best, with accuracy " << accurate << endl;
        return;
    }

    vector<int> returnFeatures()
    {
        return featuresIncluded;
    }

    void push_back(int index)
    { //Push a new feature index to this set
        featuresIncluded.push_back(index);
        return;
    }

    void setAccurate(double val)
    {
        this->accurate = val;
        return;
    }
    double getAccurate()
    {
        return accurate;
    }

    int featuresSize()
    {
        return featuresIncluded.size();
    }

    void setFeatures(vector<int> features)
    {
        for (int i = 0; i < features.size(); i++)
        {
            featuresIncluded.push_back(features.at(i));
        }
    }

    //https://www.bestprog.net/en/2019/04/27/overloading-the-assignment-operator-examples/
    /*Node operator=(vector<int> iteration)
    {
        // implementation of operator function
        // ...

        for (int i = 0; i < iteration.size(); i++)
        {
            featuresIncluded.push_back(iteration.at(i));
        }
    }*/
};

//https://www.codeguru.com/cpp/cpp/cpp_mfc/stl/article.php/c4027/C-Tutorial-A-Beginners-Guide-to-stdvector-Part-1.htm
//Each instance is a slice of the dataset- A single Row
class instance //A single instance is a row in the data set can also be seen as a point in graph
{
public:
    double getClass()
    {
        return classifier;
    }
    void setClass(double settingClass)
    {
        classifier = settingClass;
    }
    void replace(int i, double feature)
    {
        features.at(i) = feature;
    }
    void setFeature(double feature)
    {
        features.push_back(feature);
    }
    double getFeature(int position)
    {
        return features.at(position);
    }
    void printFeatures()
    {
        for (int i = 0; i < features.size(); i++)
        {
            cout << features.at(i) << "  ";
        }
        cout << endl;
    }
    int getFeatureSize()
    {
        return features.size();
    }

private:
    vector<double> features; //The features of this instance
    double classifier = -1;  // Is this instance class 0  or 1
};

double eucleadianDistance(instance testSet, instance trainingSet, vector<int> featureSet)
{ //https://en.wikipedia.org/wiki/Euclidean_distance

    //Feature Set contains the index's we want to test.
    double distance = 0;

    //Go through each feature in the instance currently being trained
    //We will only test the features currently being tested.
    for (int i = 0; i < featureSet.size(); i++)
    {
        //cout << "currently subtracting " << testSet.getFeature(featureSet.at(i)) << " from " << trainingSet.getFeature(featureSet.at(i)) << endl;
        distance += pow((testSet.getFeature(featureSet.at(i)) - trainingSet.getFeature(featureSet.at(i))), 2);
    }
    distance = sqrt(distance);
    //cout << "Distance being square rooted!" << distance << endl;
    //distance = sqrt(distance);
    //cout << "returning distance: " << distance << endl;
    return distance;
}

//Nearest Neighbor explained https://www.youtube.com/watch?v=UqYde-LULfs
double NearestNeighbor(vector<int> featureSet, vector<instance> dataSet)
{
    //We sent Nearest Neighbor the entire data set
    //We also sent the features we want to test accuracy on.

    //Iterate througheach datapoint(instance) calculate eucledian distance to each point
    //K should be odd if classifying two classes
    //K shouldn't be a multiple of number of classes
    double correct;
    double totCount;

    //cout << "-------------" << endl;

    //This will do all possible training getting one instance and testing it with the rest of the data set
    //i is the current instance that will be tested
    for (int i = 0; i < dataSet.size(); i++)
    {

        double distance = 500; //smallest distance will determine the class. Set to MAX int value so first run sets smallest ditance appropriately
        instance NN;           //Once we find the nearest neighbor set NN as that nearest neighbor

        //correct = 0;  <---------------CHECK IF CORRECT HAS TO BE HERE ISSUE IS ACCURACY

        //Test instance slice i against the rest of the dataset
        for (int j = 0; j < dataSet.size(); j++)
        {
            if (j == i)
            {
                //Do nothing if get to instance i that is currently being tested against the dataset
                continue;
            }
            else
            {
                /*
                cout << "Testing features: " << endl;
                for (int y = 0; y < featureSet.size(); y++)
                {
                    cout << featureSet.at(y) << " ";
                }
                cout << endl;

                cout << "Against Testing Set: " << i << endl;
                dataSet.at(i).printFeatures();

                cout << "Training set: " << j << endl;
                dataSet.at(j).printFeatures();
                */
                double currentDistance = eucleadianDistance(dataSet.at(i), dataSet.at(j), featureSet);

                //A new nearest neighbor is found, mark this as the closest distance to our unlabeled datapoint
                //And set the nearest neighbors class as our predicted class label to the unlabled data point
                if (currentDistance < distance)
                {
                    //cout << "------ New Nearest Neighbor found--- " << j << endl;
                    //cout << currentDistance << endl;
                    distance = currentDistance;
                    NN.setClass(dataSet.at(j).getClass()); //This is currently the closest datapoint to our testing point.  Update our predicted clas each time a closer neighbor is found
                }
            }
        } //After checking the test to every other data point should get a predicted class

        //Check if the predicted class was correct
        //Here we check our prediction(NN) with the actual class
        if (NN.getClass() == dataSet.at(i).getClass())
        {
            correct++;
        }
        totCount++;
    }

    //cout << "-------------" << endl;
    //In the end we check our classifiers accuracy that used a certain set of features
    double accuracy = (correct / dataSet.size());
    //cout << "Total Count is: " << totCount << endl;

    /*
    cout << "Correct Count is: " << correct << endl;
    cout << "Accuracy: " << accuracy << "\n\n";
    */

    //double accuracy = (correct / totCount);
    return accuracy;
}

//This function normalizes the dataset
void Normalize(vector<instance> &dataSet) //pass by refrence or change won't take effect
{
    //For each feature column, calculate the mean and standard deviation,
    //then replace each element x of that column with (x - mean)/std. Repeat for each column.

    //To find mean sum the collection of numbers / number of numbers
    vector<double> sum;
    sum.resize(dataSet.at(0).getFeatureSize(), 0);
    vector<double> mean;
    mean.resize(dataSet.at(0).getFeatureSize(), 0);
    vector<double> stdDev;
    stdDev.resize(dataSet.at(0).getFeatureSize(), 0);
    double n = 0; //number of numbers in column

    //cout << "Goin through " << dataSet.size() << " instances." << endl;

    //Iterate through the list of instances
    for (int i = 0; i < dataSet.size(); i++)
    {
        //For each instance iterate through features
        //Each column needs to be summed up
        for (int j = 0; j < dataSet.at(0).getFeatureSize(); j++)
        {
            sum.at(j) += dataSet.at(i).getFeature(j);
        }
    }

    //cout << "Mean for each column: " << endl;
    //Mean is sum of numbers / tot numbers
    //Sum of col features / datasetSize()
    for (int i = 0; i < dataSet.at(0).getFeatureSize(); i++)
    {
        mean.at(i) = (sum.at(i) / dataSet.size());
        //cout << mean.at(i) << " | ";
    }
    //cout << endl<< endl;

    //Finding the standard deviation
    //https://www.khanacademy.org/math/probability/data-distributions-a1/summarizing-spread-distributions/a/calculating-standard-deviation-step-by-step

    //With it's j mean
    //Calculate staDeviation for feature column
    for (int i = 0; i < dataSet.size(); i++)
    {
        //For each instance iterate through features
        //Each column needs to be summed up
        for (int j = 0; j < dataSet.at(0).getFeatureSize(); j++)
        {
            stdDev.at(j) += (pow(abs(dataSet.at(i).getFeature(j) - mean.at(j)), 2)) / dataSet.size();
        }
    }

    //Display stdDev for each column

    //cout << "Standard Deviation is for each column is:" << endl;
    for (int i = 0; i < stdDev.size(); i++)
    {
        stdDev.at(i) = sqrt(stdDev.at(i));
        //cout << stdDev.at(i) << " | ";
    }
    //cout << endl << endl;

    //Replace each value x in the dataset
    //Iterate through the list of instances
    //Replace each feature at position (j)
    for (int i = 0; i < dataSet.size(); i++)
    {
        //For each instance iterate through features
        //replace the value in that feature by (x-mean)/std deviation
        for (int j = 0; j < dataSet.at(0).getFeatureSize(); j++)
        {
            dataSet.at(i).replace(j, ((dataSet.at(i).getFeature(j) - mean.at(j)) / stdDev.at(j)));
        }
    }
}

//Function returns a vector full of the dataset
//https://stackoverflow.com/questions/7880/how-do-you-open-a-file-in-c
//https://stackoverflow.com/questions/20739453/using-getline-with-file-input-in-c
//https://stackoverflow.com/questions/1710447/string-in-scientific-notation-c-to-double-conversion //NORMALIZATION?
//https://stackoverflow.com/questions/9411118/convert-scientific-notation-to-decimal-in-c
vector<instance> getData(string fileName)
{
    ifstream inFile;
    inFile.open(fileName.c_str());
    if (!inFile.is_open())
    {
        cout << "Error opening file";
    }
    vector<instance> dataSet; //Data set that we will return
    string line;
    string temp;
    string classHold;
    string feature;
    while (getline(inFile, line))
    {
        instance row; // During each line create a new instance node and save the line data in it, put it in the data set vector.

        stringstream streamLine(line); //Input stream class to operate on strings.
        //cout << "SS line: " << line << endl;
        // streamLine >> see;
        //cout << "IN SEE: " << see << endl;
        streamLine >> classHold; //The first value will always be the class
                                 //cout << "Hold: " << classHold << endl;
                                 //cout << "Class before placed in instance: " << classHold << endl;
                                 //classHold = atof(temp);
                                 //cout << "TEMP = " << temp << endl;
                                 //cout << "CLASS HOLD=" << classHold << endl;

        row.setClass(stod(classHold));
        while (streamLine >> feature)
        { //Parse the entire row, continue until end of line reached.
            //cout << "Features before placed in instance: " << feature << endl;
            row.setFeature(stod(feature));
        }
        dataSet.push_back(row);
    }

    inFile.close();
    /*
    for (int i = 0; i < dataSet.size(); i++)
    {
        dataSet.at(i).printFeatures();
    }*/

    return dataSet;
}

struct Manhatten
{
    //Overide
    bool operator()(const Node lhs, const Node rhs)
    {

        //reuse comparator code from eight puzzle
        return (lhs.accurate < rhs.accurate);
    }
};

/*
*/
// This function finds classification of point p using
// k nearest neighbour algorithm. It assumes only two
// groups and returns 0 if p belongs to group 0, else
// 1 (belongs to group 1).

/*
1) Nearest Neighbor classifier
-Keep training instances in memory(put them in an array)
-When new data point (leftOut) is given,compute distance to the training points
-Return class label of nearest training point
*/

//Not hasn't been passed by refence... check on later
void fowardSelection(vector<instance> dataSet)
{
    //http: //www.cplusplus.com/reference/queue/priority_queue/priority_queue/
    priority_queue<Node, std::vector<Node>, Manhatten> greedyFeaturesQueue; //As we go foward the greedy search will get the node with the best accuracy
    int accuracyMax;                                                        //Holds current best accuracyScore

    Node max;
    max.setAccurate(0);
    //Will hold the features
    Node temp;

    //This is the first node with an empty set of features.
    //temp.setAccurate(NearestNeighbor(temp.returnFeatures(), dataSet));
    //temp.printFeatures();

    vector<int> push;
    //push.resize(dataSet.at(0).getFeatureSize(), 0);

    //We will continue, add a feature, select the node with the best accuracy, add a feature and continue

    cout << "DATASET FEATURES SIZE " << dataSet.at(0).getFeatureSize() << endl;

    //double FREE ERROR, deleting something twice in queue
    ///https://stackoverflow.com/questions/14063791/double-free-or-corruption-after-queuepush

    for (int i = 0; i < dataSet.at(0).getFeatureSize(); i++) //For each of the 10 features, make a node
    {

        for (int j = 0; j < dataSet.at(0).getFeatureSize(); j++)
        {
            bool check = false;
            vector<int> features; //A new node is created each time
            double accuracyScore;
            features = push; //Overload assignment operator add extra features.
            //cout << "About to start entering data" << endl;

            //Big if check looking for repeated features
            //This just sets a flag to find accuracy of current features in features vector
            //If current feature is already in features vector skip trying to test against classifier
            for (int l = 0; l < features.size(); l++)
            {
                if (j == features.at(l))
                {                 //Iterate through entire list of features make sure current feature index not in
                    check = true; //Already in list of features skip this step
                }
            }

            //This check makes sure we don't put in same feature index twice
            //set feature pushes back to vector, it doesn't replace
            if (check == false)
            {
                //cout << "----------------------\n\n";
                features.push_back(j); //Pushing back the index of the features we are testing

                //Display the current node of features being tested
                cout << "\tCurrent Node Features: " << endl;
                cout << "\t";
                for (int x = 0; x < features.size(); x++)
                {
                    cout << features.at(x) << " ";
                }
                cout << endl;

                //cout << "Entering Nearest Neighbor" << endl;
                accuracyScore = NearestNeighbor(features, dataSet);
                Node tempSet; //Make node here to avoid double corruption issues
                tempSet.setFeatures(features);
                cout << "\tAccuracy is " << accuracyScore << endl;
                tempSet.setAccurate(accuracyScore);
                //cout << "Level " << i << " Node: ";
                //tempSet.printFeatures();
                greedyFeaturesQueue.push(tempSet);
                //cout << "----------------------\n\n";

                // cout << "In iteration " << i << ", " << j << " the features currently in the vector are ";
                //features.printFeatures();
            }
        }

        temp = greedyFeaturesQueue.top();

        cout << endl;
        cout << "Best Feature Set and Accuracy to Expand" << endl;
        temp.printFeatures();
        cout << endl;

        if (temp.getAccurate() > max.getAccurate())
        {
            max = temp;
            cout << "New Max Accuracy Found** ";
            max.printFeatures();
            cout << "***\n";
        }
        /*
        cout << "MAX Node is: " << endl;
        max.printFeatures();
        */

        push = temp.featuresIncluded; //<-------------Problem Might be here!!!! When do we push the new feature that has best accuracy

        //Only want best accuracy for the current expanded row not from entire node tree
        while (!greedyFeaturesQueue.empty()) //clear entire queue, only concerned with highest %
        {
            //cout << "Clearing QUEUE" << endl;
            //cout << greedyFeaturesQueue.top().accurate << " | " << endl;
            greedyFeaturesQueue.pop();
        }
    }

    cout << "Best features are ";
    max.printFeatures();
}

//Backward selection instead of building up the best possible feature set
//removes features to acheive the best feature set
void backwardElimination(vector<instance> dataSet)
{
    //As we remove nodes we will still want to keep a list of the nodes
    //With the best accuracy.
    priority_queue<Node, std::vector<Node>, Manhatten> greedyFeaturesQueue;

    int accuracyMax = 0; //Holds current best accuracyScore
    int position = 0;
    Node max;
    max.setAccurate(0);
    //Will hold the features
    Node temp;

    vector<int> push;

    //The features that are going to be pushed should be same size as the feature set in the instances
    for (int i = 0; i < dataSet.at(0).getFeatureSize(); ++i)
    {
        push.push_back(i);
        cout << push.at(i) << endl;
    }

    cout << "DATASET FEATURES SIZE " << dataSet.at(0).getFeatureSize() << endl;
    cout << "Initial Start is: " << endl;
    for (int i = 0; i < push.size(); i++)
    {
        cout << push.at(i) << " ";
    }
    cout << endl;

    max.setAccurate(NearestNeighbor(push, dataSet));
    max.featuresIncluded = push;

    max.printFeatures();

    for (int i = 0; i < dataSet.at(0).getFeatureSize(); i++) //For each of the 10 features, make a node
    {

        for (int j = 0; j < dataSet.at(0).getFeatureSize(); j++)
        {

            bool check = false;
            vector<int> features; //A new node is created each time
            double accuracyScore;
            features = push; //Overload assignment operator add extra features.
            //cout << "About to start entering data" << endl;
            //cout << "Made it before check" << endl;
            //Big if check looking for repeated features
            //This just sets a flag to find accuracy of current features in features vector
            //If current feature is already in features vector skip trying to test against classifier

            for (int l = 0; l < features.size(); l++)
            {
                //cout << "Check if " << l << "is in feature set " << features.at(l) << endl;
                if (j == features.at(l)) //Only remove features if they
                {                        //Iterate through entire list of features make sure current feature index not in

                    //cout << "IT IS" << endl;
                    check = true; //If in list then we want to remove it
                    position = l;
                }
            }

            //This check is to see if j is in feature list so we can remove it
            if (check == true)
            {
                //cout << "Removing J from node" << j << endl;
                //cout << "----------------------\n\n";
                //instead of pushing a feature, remove feature j
                //cout << "Erasing feature: " << j << endl;
                features.erase(features.begin() + position); //Delete one value at a  time
                //features.push_back(j); //Pushing back the index of the features we are testing
                //Display the current node of features being tested
                cout << "\tCurrent Node Features: " << endl;
                cout << "\t";
                for (int x = 0; x < features.size(); x++)
                {
                    cout << features.at(x) << " | ";
                }
                cout << endl;

                //cout << "Entering Nearest Neighbor" << endl;
                accuracyScore = NearestNeighbor(features, dataSet);
                Node tempSet; //Make node here to avoid double corruption issues
                tempSet.setFeatures(features);
                cout << "\tAccuracy is " << accuracyScore << "\n\n";
                tempSet.setAccurate(accuracyScore);
                //cout << "Level " << i << " Node: ";
                //tempSet.printFeatures();
                greedyFeaturesQueue.push(tempSet);
                //cout << "----------------------\n\n";

                // cout << "In iteration " << i << ", " << j << " the features currently in the vector are ";
                //features.printFeatures();
            }
        }

        temp = greedyFeaturesQueue.top();

        cout << endl;
        cout << "Best Feature Set and Accuracy to Expand" << endl;
        temp.printFeatures();

        if (temp.getAccurate() > max.getAccurate())
        {
            max = temp;
            cout << "**New Max Accuracy Found** ";
            max.printFeatures();
            cout << "\n";
        }
        /*
        cout << "MAX Node is: " << endl;
        max.printFeatures();
        */

        push = temp.featuresIncluded; //<-------------Problem Might be here!!!! When do we push the new feature that has best accuracy

        //Only want best accuracy for the current expanded row not from entire node tree
        while (!greedyFeaturesQueue.empty()) //clear entire queue, only concerned with highest %
        {
            //cout << "Clearing QUEUE" << endl;
            //cout << greedyFeaturesQueue.top().accurate << " | " << endl;
            greedyFeaturesQueue.pop();
        }
    }

    cout << "Best features are ";
    max.printFeatures();
}

int main()
{
    string fileName;
    cout << "Welcome to Cristian's Feature Selection Algorithim." << endl;
    cout << "Type in the name of the file to test" << endl;
    cin >> fileName;
    cout << "Opening file... " << fileName << endl;

    //getData gets the entire data set, normalizes and puts it in the dataSet vector
    //Make sure to get each row into the dataSet
    vector<instance> dataSet = getData(fileName);

    cout << "Successfully retrieved data." << endl;

    //Data points aka rows
    int numbInstances = dataSet.size();

    //Columns excluding the class column
    int numbFeatures = dataSet.at(0).getFeatureSize();

    //Keep track of how good classifier is
    int counter = 0;

    cout << "\n\n";

    //Print The Entire Data Set
    /*
    cout << "---Printing Data Set---" << endl;
    for (int i = 0; i < dataSet.size(); i++)
    {
        cout << "Line: " << i << endl;
        cout << "Class: " << dataSet.at(i).getClass() << " Features: ";
        dataSet.at(i).printFeatures();
    }
    */

    cout << "The dataset has " << numbFeatures << " features (not including the class attribute) with " << numbInstances << " instances\n";
    cout << "Normalizing data ...\n";
    Normalize(dataSet);

    //Print The Entire Normalized Data Set
    /*
    cout << "---Printing Normalized Data Set---" << endl;
    for (int i = 0; i < dataSet.size(); i++)
    {
        cout << "Line: " << i << endl;
        cout << "Class: " << dataSet.at(i).getClass() << " Features: ";
        dataSet.at(i).printFeatures();
    }
    */
    int choice = 0;
    while (choice <= 0 || choice > 3)
    {
        cout << endl;
        cout << "Type the number of algorithim you want to run\n";
        cout << "1) foward selection\n";
        cout << "2) backward selection\n";
        cout << "3) special....\n";
        cin >> choice;
    }

    switch (choice)
    {
    case 1:
        cout << "Selected Foward Selection\n";
        //cout << "Training Classifier...\n";
        //NearestNeighbor(dataSet);
        cout << "Applying Foward Selection...\n";
        fowardSelection(dataSet);
        //Train Classifier
        //Then Test using Foward Selection
        //double total = counter / numbInstances;
        break;
    case 2:
        cout << "Selected Backward Selection\n";
        //Train Classifier
        //Then test using Backward Selection
        backwardElimination(dataSet);
        break;
    case 3:
        cout << "Selected special algorithim\n";
        break;
    }

    return 0;
}