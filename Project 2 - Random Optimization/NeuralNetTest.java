import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import shared.filt.LabelSplitFilter;
import shared.reader.CSVDataSetReader;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

public class NeuralNetTest implements Runnable {
    private static Instance[] instances;
    private static Instance[] instancesTest;

    private static int inputLayer = 13, hiddenLayer = 5, outputLayer = 1, trainingIterations = 100;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static int[] iterations = {1000, 2000, 4000, 8000, 16000};
    int[] numberOfRestarts = {1, 3, 5, 10, 15, 20};
    
    double[] coolingrates = {1E10, 1E11, 1E12, 1E13, 1E14, 1E15};
    int[] temps = {50, 55, 60, 65, 70, 75, 80, 85, 90, 95};
    
    int[] populations = {60, 120};
    int[] mates = {20, 50};
    int[] mutates = {0, 10};
    
    private static ErrorMeasure measure = new SumOfSquaresError();
    
    private static int numAttrs = 13;

    private static DataSet set; // = new DataSet(instances);
    private static DataSet setTest; // = new DataSet(instances);
   
    private static int datasetSize;

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    //private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public NeuralNetTest(int iterations, DataSet trainset, DataSet testset, Integer DSsize) {
    	this.trainingIterations = iterations;
    	this.instances = trainset.getInstances();
    	this.instancesTest = testset.getInstances();
    	this.set = trainset;
    	this.setTest = testset;
    	this.datasetSize = DSsize;
    }
    
    /*public static void main(String[] args) {
    	(new NeuralNetTest(100)).run();
    }*/
    
    
    public void run() {	
    }
    
    
    /*
    
    public void run() {		
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }
        BufferedWriter bw = null;
        try {
			bw = new BufferedWriter(new FileWriter(new File("nn_results_" + trainingIterations+".txt")));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        double[] errors = {}; 
        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, correctTest = 0, incorrectTest = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            System.out.println(oa[i].toString());

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            //Train error
            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            
            //Test error
            double predictedTest, actualTest;
            start = System.nanoTime();
            for(int j = 0; j < instancesTest.length; j++) {
                networks[i].setInputValues(instancesTest[j].getData());
                networks[i].run();

                predictedTest = Double.parseDouble(instancesTest[j].getLabel().toString());
                actualTest = Double.parseDouble(networks[i].getOutputValues().toString());

                double trashTest = Math.abs(predicted - actual) < 0.5 ? correctTest++ : incorrectTest++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);
            
            System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
            System.out.println("Test Error: " + df.format(100 - (correctTest/(correctTest+incorrectTest)*100)));

            /*results += "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            
            
            try {
				bw.write("\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
				            "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
				            + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
				            + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n");
				bw.newLine();
				bw.flush();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

            
            
        }
        try {
			bw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        System.out.println("Neural Networks for " + trainingIterations + " trained.");
    }
    
    
    
    
    */
    
    public void runRHC() throws IOException {
    	ArrayList errortrain = new ArrayList<Double>();
    	ArrayList errortest = new ArrayList<Double>();
    	
    	final String FILENAME = "RHC_" + String.valueOf(this.datasetSize) + "_" + String.valueOf(this.trainingIterations) +".csv";
    	System.out.println("RHC with dataset size:" + String.valueOf(this.datasetSize) + ", training iterations: " + String.valueOf(this.trainingIterations));
    	
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }
        BufferedWriter bw = null;
        try {
			bw = new BufferedWriter(new FileWriter(new File(FILENAME)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        for (int numberOfRestart : numberOfRestarts) {

	        oa[0] = new RandomizedHillClimbing(nnop[0]);
	        
	            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, correctTest = 0, incorrectTest = 0;
	            train(oa[0], networks[0], oaNames[0]); //trainer.train();
	            end = System.nanoTime();
	            trainingTime = end - start;
	            trainingTime /= Math.pow(10,9);
	            System.out.println(oa[0].toString());
	
	            Instance optimalInstance = oa[0].getOptimal();
	            networks[0].setWeights(optimalInstance.getData());
	
	            //Train error
	            double predicted, actual;
	            start = System.nanoTime();
	            for(int j = 0; j < instances.length; j++) {
	                networks[0].setInputValues(instances[j].getData());
	                networks[0].run();
	
	                predicted = Double.parseDouble(instances[j].getLabel().toString());
	                actual = Double.parseDouble(networks[0].getOutputValues().toString());
	
	                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
	
	            }
	            
	            //Test error
	            double predictedTest, actualTest;
	            start = System.nanoTime();
	            for(int j = 0; j < instancesTest.length; j++) {
	                networks[0].setInputValues(instancesTest[j].getData());
	                networks[0].run();
	
	                predictedTest = Double.parseDouble(instancesTest[j].getLabel().toString());
	                actualTest = Double.parseDouble(networks[0].getOutputValues().toString());
	
	                double trashTest = Math.abs(predictedTest - actualTest) < 0.5 ? correctTest++ : incorrectTest++;
	
	            }
	            end = System.nanoTime();
	            testingTime = end - start;
	            testingTime /= Math.pow(10,9);
	            
	            double trainerror = 100 - (correct/(correct+incorrect)*100);
	            double testerror = 100 - (correctTest/(correctTest+incorrectTest)*100);
	            errortrain.add(trainerror);
	            errortest.add(testerror);
	            
	            System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
	            System.out.println("Test Error: " + df.format(100 - (correctTest/(correctTest+incorrectTest)*100)));
	           
	        
	        }
        
	        for (int k = 0; k < errortrain.size(); k++) {
	            bw.write (String.valueOf(errortrain.get(k) + "," + errortest.get(k)));
	            bw.newLine();
	        }
        
	        try {
				bw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	        System.out.println("Neural Networks for " + trainingIterations + " trained.");
    }
    
    
    
    
    public void runSA() throws IOException {
    	ArrayList organizedData = new ArrayList<>();
    	
    	ArrayList errortrain = new ArrayList<Double>();
    	ArrayList errortest = new ArrayList<Double>();
    	
    	final String FILENAME = "SA_" + String.valueOf(this.datasetSize) + "_" + String.valueOf(this.trainingIterations) +".csv";
    	System.out.println("SA with dataset size:" + String.valueOf(this.datasetSize) + ", training iterations: " + String.valueOf(this.trainingIterations));
    	
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }
        BufferedWriter bw = null;
        try {
			bw = new BufferedWriter(new FileWriter(new File(FILENAME)));
		} catch (IOException e) {
			e.printStackTrace();
		}
        
        for (double coolingrate : coolingrates) {
        	for (int temp : temps) {

        		double temp_network = temp / 100;
        		oa[1] = new SimulatedAnnealing(coolingrate, temp_network, nnop[1]);
	        
	            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, correctTest = 0, incorrectTest = 0;
	            train(oa[1], networks[1], oaNames[1]); //trainer.train();
	            end = System.nanoTime();
	            trainingTime = end - start;
	            trainingTime /= Math.pow(10,9);
	            System.out.println(oa[1].toString());
	
	            Instance optimalInstance = oa[1].getOptimal();
	            networks[1].setWeights(optimalInstance.getData());
	
	            //Train error
	            double predicted, actual;
	            start = System.nanoTime();
	            for(int j = 0; j < instances.length; j++) {
	                networks[1].setInputValues(instances[j].getData());
	                networks[1].run();
	
	                predicted = Double.parseDouble(instances[j].getLabel().toString());
	                actual = Double.parseDouble(networks[1].getOutputValues().toString());
	
	                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
	
	            }
	            
	            //Test error
	            double predictedTest, actualTest;
	            start = System.nanoTime();
	            for(int j = 0; j < instancesTest.length; j++) {
	                networks[1].setInputValues(instancesTest[j].getData());
	                networks[1].run();
	
	                predictedTest = Double.parseDouble(instancesTest[j].getLabel().toString());
	                actualTest = Double.parseDouble(networks[1].getOutputValues().toString());
	
	                double trashTest = Math.abs(predictedTest - actualTest) < 0.5 ? correctTest++ : incorrectTest++;
	
	            }
	            end = System.nanoTime();
	            testingTime = end - start;
	            testingTime /= Math.pow(10,9);
	            
	            double trainerror = 100 - (correct/(correct+incorrect)*100);
	            double testerror = 100 - (correctTest/(correctTest+incorrectTest)*100);
	            errortrain.add(trainerror);
	            errortest.add(testerror);
	            
	            System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
	            System.out.println("Test Error: " + df.format(100 - (correctTest/(correctTest+incorrectTest)*100)));
	           
	        
	        }
        }
        
	        for (int k = 0; k < errortrain.size(); k++) {
	            bw.write (String.valueOf(errortest.get(k)));
	            bw.newLine();
	            bw.write (String.valueOf(errortrain.get(k)));
	            bw.newLine();
	        }
        
	        try {
				bw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	        System.out.println("Neural Networks for " + trainingIterations + " trained.");
    }
    
    
    public void runGA() throws IOException {
    	ArrayList organizedData = new ArrayList<>();
    	
    	ArrayList errortrain = new ArrayList<Double>();
    	ArrayList errortest = new ArrayList<Double>();
    	
    	final String FILENAME = "GA_" + String.valueOf(this.datasetSize) + "_" + String.valueOf(this.trainingIterations) +".csv";
    	System.out.println("GA with dataset size:" + String.valueOf(this.datasetSize) + ", training iterations: " + String.valueOf(this.trainingIterations));
    	
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }
        BufferedWriter bw = null;
        try {
			bw = new BufferedWriter(new FileWriter(new File(FILENAME)));
		} catch (IOException e) {
			e.printStackTrace();
		}
        
        for (int pop : populations) {
        	for (int mate : mates) {
        		for (int mutate : mutates) {
        	
        		oa[2] = new StandardGeneticAlgorithm(pop, mate, mutate, nnop[2]);
	        
	            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, correctTest = 0, incorrectTest = 0;
	            train(oa[2], networks[2], oaNames[2]); //trainer.train();
	            end = System.nanoTime();
	            trainingTime = end - start;
	            trainingTime /= Math.pow(10,9);
	            System.out.println(oa[2].toString());
	
	            Instance optimalInstance = oa[2].getOptimal();
	            networks[2].setWeights(optimalInstance.getData());
	
	            //Train error
	            double predicted, actual;
	            start = System.nanoTime();
	            for(int j = 0; j < instances.length; j++) {
	                networks[2].setInputValues(instances[j].getData());
	                networks[2].run();
	
	                predicted = Double.parseDouble(instances[j].getLabel().toString());
	                actual = Double.parseDouble(networks[2].getOutputValues().toString());
	
	                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
	
	            }
	            
	            //Test error
	            double predictedTest, actualTest;
	            start = System.nanoTime();
	            for(int j = 0; j < instancesTest.length; j++) {
	                networks[2].setInputValues(instancesTest[j].getData());
	                networks[2].run();
	
	                predictedTest = Double.parseDouble(instancesTest[j].getLabel().toString());
	                actualTest = Double.parseDouble(networks[2].getOutputValues().toString());
	
	                double trashTest = Math.abs(predictedTest - actualTest) < 0.5 ? correctTest++ : incorrectTest++;
	
	            }
	            end = System.nanoTime();
	            testingTime = end - start;
	            testingTime /= Math.pow(10,9);
	            
	            double trainerror = 100 - (correct/(correct+incorrect)*100);
	            double testerror = 100 - (correctTest/(correctTest+incorrectTest)*100);
	            errortrain.add(trainerror);
	            errortest.add(testerror);
	            
	            System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
	            System.out.println("Test Error: " + df.format(100 - (correctTest/(correctTest+incorrectTest)*100)));
	         
        		}
        	}
        }
        
	        for (int k = 0; k < errortrain.size(); k++) {
	            bw.write (String.valueOf(errortest.get(k)));
	            bw.newLine();
	            bw.write (String.valueOf(errortrain.get(k)));
	            bw.newLine();
	        }
        
	        try {
				bw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	        System.out.println("Neural Networks for " + trainingIterations + " trained.");
    }
    

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

//            System.out.println(df.format(error));
        }
    }

    /*private static Instance[] initializeInstances() {

        double[][][] attributes = new double[151][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("data/hd_train.fann")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[numAttrs]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < numAttrs; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }*/
}
