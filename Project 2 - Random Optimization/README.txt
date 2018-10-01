CS 4641 - Machine Learning
Project 2 - Randomized Optimization
Nilay Mehta, 903254155

The datasets I've used for my project are downloaded and included in my submission, but their original links can be found here:

Human Resources Analytics(HR)
https://www.kaggle.com/ludobenistant/hr-analytics/data

I used ABAGAIL and Java SDK 8 to run all of the algorithms for both parts of this project.

For the sake of convenience, I have included a the folder source that includes my eclipse project used to run all the experiments along with the requested code in the root of the submission.

The file filterHR.py was used to sample the dataset and split the dataset into train and test datasets.

Installation for ABAGAIL (pulled from original ABAGAIL repo):
1. Install Java 8 SDK from here http://www.oracle.com/technetwork/java/javase/downloads/index.html
2. Install Ant http://ant.apache.org/
3. Clone or download source files from Git
4. Go with command line to where the build.xml file is and run: ant (note: the ant executable should be in your path somehow if you installed ant correctly.. so will java and javac)
5. Now run your scripts

To run the code:
1. Start a project in Eclipse with Java 1.8
2. link the abagail-lib.jar to the compile path
3. Download the ABAGAIL library from Github using instructions above
5. Copy the datasets folder (and results folder if you would like) to the source of your project (same level as your src folder)
6. Copy all .java files into the src folder of your project
7.The main file to run the experiments are in src/ProjectRunner.java, and the main function is listed below:


    public static void main(String[] args) throws IOException, Exception {
        randomrestartRHC();               //random restarts for randomized hill climbing
        SA();                             //Simulated Annealing
        GA();                             //Genetic Algorithm
        TravelingSalesman.main(null);     //Optimization Problem 1
        FlipFlop.main(null);              //Optimization Problem 2
        FourPeaks.main(null);             //Optimization Problem 3
    }

To run any one particular experiment, you can comment the other methods out.