import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.Instance;

/**
 * A test using the 4 Peaks evaluation function
 * @author James Liu
 * @version 1.0
 */
public class FourPeaks {

    public static void main(String[] args) {
    	System.out.println("---------- Four Peaks ----------");
        int N = 100;
        int T = 8;
        int iterations = 50;
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        System.out.println("Randomized Hill Climbing\n---------------------------------");
        for(int i = 0; i < iterations; i++)
        {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 15000);
            fit.train();
            System.out.println(ef.value(rhc.getOptimal()) + ", " + (((double)(50)*(System.nanoTime() - t))/ 1e9d));
        }

        System.out.println("Simulated Annealing \n---------------------------------");
        for(int i = 0; i < iterations; i++)
        {
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(sa, 15000);
            fit.train();
            System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(50)*(System.nanoTime() - t))/ 1e9d));
        }

        System.out.println("Genetic Algorithm\n---------------------------------");
        for(int i = 0; i < iterations; i++)
        {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(50)*(System.nanoTime() - t))/ 1e9d));
        }

        System.out.println("MIMIC \n---------------------------------");
        for(int i = 0; i < iterations; i++)
        {
            MIMIC mimic = new MIMIC(100, 5, pop);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            System.out.println(ef.value(mimic.getOptimal()) + ", " + (((double)(50)*(System.nanoTime() - t))/ 1e9d));
        }
    }
}