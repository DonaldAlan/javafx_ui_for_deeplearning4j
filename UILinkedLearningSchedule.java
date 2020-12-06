package org.deeplearning4j.examples.wip.javafxui;

import org.nd4j.linalg.schedule.ISchedule;

import java.io.Serializable;
import java.text.NumberFormat;

/**
 * @author Donald A. Smith (ThinkerFeeler@gmail.com)
 *
 * The logic for updating the public static double learningRate is in the linked class TrainingListenerWithUI
 */
public class UILinkedLearningSchedule implements ISchedule, Serializable, Cloneable {
    private static UILinkedLearningSchedule instance = new UILinkedLearningSchedule();
    public static double learningRate = 0.001;
    private static final NumberFormat numberFormat = NumberFormat.getInstance();
    private final String name;
    static {
        numberFormat.setMaximumFractionDigits(7);
    }
    private UILinkedLearningSchedule() {
        String name= this.toString();
        int index=name.indexOf('@');
        this.name = name.substring(index+1, name.length());
        System.out.println("Calling constructor of MusicLearningSchedule, yielding " + name);
    }
    public static UILinkedLearningSchedule getInstance(double learningRate) {
        UILinkedLearningSchedule.learningRate = learningRate;
        return instance;
    }
    @Override
    public double valueAt(int iteration, int epoch) {
        return learningRate;
    }

    @Override
    public ISchedule clone() {
        System.out.println("Calling clone");
        return this;
    }
}
