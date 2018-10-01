import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
// import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesman {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
    	System.out.println("---------- Traveling Salesman ----------");
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        double startrhc, trainingTimerhc, endrhc = 0;
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainerMod fit = new FixedIterationTrainerMod(rhc, 15000);
        startrhc = System.nanoTime();
        fit.train(startrhc);
        endrhc = System.nanoTime();
        trainingTimerhc = endrhc - startrhc;
        trainingTimerhc /= Math.pow(10,9);

        System.out.println("RHC: " + ef.value(rhc.getOptimal()) + " Time: " + trainingTimerhc);
        
        double startsa, trainingTimesa, endsa = 0;

        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        fit = new FixedIterationTrainerMod(sa, 15000);
        startsa = System.nanoTime();
        fit.train(startsa);
        endsa = System.nanoTime();
        trainingTimesa = endsa - startsa;
        trainingTimesa /= Math.pow(10,9);

        System.out.println("SA: " + ef.value(sa.getOptimal()) + " Time: " + trainingTimesa);
        
        double startga, trainingTimega, endga = 0;

        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
        fit = new FixedIterationTrainerMod(ga, 2000);
        startga = System.nanoTime();
        fit.train(startga);
        endga = System.nanoTime();
        trainingTimega = endga - startga;
        trainingTimega /= Math.pow(10,9);

        System.out.println("GA: " + ef.value(ga.getOptimal()) + " Time: " + trainingTimega);
        
        double startm, trainingTimem, endm = 0;

        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainerMod(mimic, 1000);
        startm = System.nanoTime();
        fit.train(startm);
        endm = System.nanoTime();
        trainingTimem = endm - startm;
        trainingTimem /= Math.pow(10,9);

        System.out.println("RHC: " + ef.value(rhc.getOptimal()) + " Time: " + trainingTimerhc);
        System.out.println("SA: " + ef.value(sa.getOptimal()) + " Time: " + trainingTimesa);
        System.out.println("GA: " + ef.value(ga.getOptimal()) + " Time: " + trainingTimega);
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()) + " Time: " + trainingTimem);

    }
}