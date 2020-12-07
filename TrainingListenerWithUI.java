package org.deeplearning4j.examples.wip.javafxui;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.control.Tooltip;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.util.Duration;
import org.deeplearning4j.core.storage.Persistable;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.model.stats.api.StatsType;
import org.deeplearning4j.ui.model.stats.impl.SbeStatsReport;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;


/**
 * @author Donald A. Smith (ThinkerFeeler@gmail.com)
 *
 * Linked with UILinkedLearningSchedule
 */
public class TrainingListenerWithUI extends Application implements TrainingListener {
    private final NumberFormat numberFormat1 = NumberFormat.getInstance();
    private final NumberFormat numberFormat2 = NumberFormat.getInstance();
    private final NumberFormat numberFormat3 = NumberFormat.getInstance();
    private final NumberFormat numberFormat6 = NumberFormat.getInstance();
    private final NumberFormat numberFormat7 = NumberFormat.getInstance();
    private final NumberFormat numberFormatForMinus = new DecimalFormat(" 0.00;-0.00");
    private long countOfForwardPasses = 0;

    private static final int WIDTH = 700;
    private static final int HEIGHT = 300;
    private Group root;
    private Label epochLabel;
    private Label epochLabelLabel;
    private Label bestScoreLabelLabel;
    private Label bestScoreLabel;
    private Label scoreLabelLabel; // Holds "Score: "
    private Label scoreLabel; // Holds the changing score
    private Label learningRateDecayLabel;
    private Label learningRateDecayLabelLabel;
    private Label learningRateLabelLabel;
    private Label learningRateLabel;
    private Label activationsLabel;
    private Label ratiosLabel;
    private Label weightsLabel;
    private Text activationsText;
    private Label gradientsTextLabel;
    private Text gradientsStdText;
    private Text ratioText;
    private Text weightsText;
    private Label numberOfParametersLabelLabel;
    private Label numberOfParametersLabel;
    private long lastUpdateMlsForScore = 0;
    private long lastUpdateMlsForGradients = 0;
    private double bestScore = Double.MAX_VALUE;
    private long bestTime = 0;
    public static TrainingListenerWithUI instance;
    private double learningRateDecayFactorPerGradientCalculation = 0.99995;
    private double baseLearningRate = 0.001;
    private StatsStorage statsStorage;
    private long startTimeInMillseconds = System.currentTimeMillis();
    private final List<double[]> scores = Collections.synchronizedList(new ArrayList<>());

    private Label runtimeLabelLabel;
    private Label runtimeLabel;
    private ScoresStage scoresStage;

    //----------------------------------------------------------------
    public TrainingListenerWithUI() {
        numberFormat1.setMinimumFractionDigits(1);
        numberFormat1.setMaximumFractionDigits(1);
        numberFormat2.setMinimumFractionDigits(2);
        numberFormat2.setMaximumFractionDigits(2);
        numberFormat3.setMinimumFractionDigits(3);
        numberFormat3.setMaximumFractionDigits(3);
        numberFormat6.setMaximumFractionDigits(6);
        numberFormat7.setMaximumFractionDigits(7);
        instance = this;
    }


    private class ScoresStage extends Stage {
        private int width=900;
        private int height= 600;
        private final LineChart<Number,Number> lineChart;
        private final Group root = new Group();
        private final  XYChart.Series series = new XYChart.Series();
        private double maxScore = 0.0;
        private int lastScoresLength=0;
        private double minScore = Double.MAX_VALUE;
        private final Label smoothLabel = new Label("Smoothing:");
        private final Spinner<Integer> smoothSpinner;
        private final NumberAxis xAxis = new NumberAxis();
        private final NumberAxis yAxis = new NumberAxis();
        private final Scene scene;
        private void handleKeyEvent(KeyEvent keyEvent) {
            if (scores.size()<5) {
                System.out.println("Returning because scores.size()= " + scores);
                return;
            }
            KeyCode code = keyEvent.getCode();
            lineChart.requestFocus();
            double f = keyEvent.isShiftDown()? 1.5: 1.1;
            switch (code) {
                case ESCAPE:
                    xAxis.setAutoRanging(true);
                    yAxis.setAutoRanging(true);
                    break;
                case UP: // show more
                    if (!yAxis.isAutoRanging()) {
                        yAxis.setUpperBound(f * yAxis.getUpperBound());
                        if (yAxis.getUpperBound() >= 1.1 * maxScore) {
                            yAxis.setAutoRanging(true);
                        }
                    }
                    break;
                case DOWN:
                    if (yAxis.isAutoRanging()) {
                        yAxis.setAutoRanging(false);
                        yAxis.setUpperBound(maxScore / f);
                    } else {
                        yAxis.setUpperBound(yAxis.getUpperBound() / f);
                    }
                    break;
                case LEFT: // show more
                    if (!xAxis.isAutoRanging()) {
                        double lastTime = scores.get(scores.size() - 1)[0];
                        double delta = keyEvent.isShiftDown() ? 0.5 * lastTime : 0.1 * lastTime;
                        double newLowerBound = Math.max(0, xAxis.getLowerBound() - delta);
                        xAxis.setLowerBound(newLowerBound);
                        if (newLowerBound == 0) {
                            xAxis.setAutoRanging(true);
                        }
                    }
                    break;
                case RIGHT: { // show less
                    final double lastTime = scores.get(scores.size() - 1)[0];
                    if (xAxis.isAutoRanging()) {
                        final double delta = keyEvent.isShiftDown() ? 0.5 * lastTime : 0.1 * lastTime;
                        xAxis.setAutoRanging(false);
                        xAxis.setLowerBound(delta);
                    } else {
                        //xLowerRange = xLowerRange + 0.1*(xAxis.getUpperBound()-xLowerRange);
                        double oldLowerBound = xAxis.getLowerBound();
                        double showingBound = lastTime - oldLowerBound;
                        final double delta = keyEvent.isShiftDown() ? 0.5 * showingBound : 0.1 * showingBound;
                        xAxis.setLowerBound(oldLowerBound + delta);
                    }
                }
                    break;
                case S: { // make smoother. Increase spinner
                    int delta = keyEvent.isShiftDown() ? 10 : 1;
                    int newValue = delta + smoothSpinner.getValue();
                    if (newValue> scores.size()/2) {
                        newValue = scores.size()/2;
                    }
                    smoothSpinner.getValueFactory().setValue(newValue);
                    lastScoresLength = 0;
                    build();
                }
                    break;
                case U: {
                    int delta = keyEvent.isShiftDown() ? -10 : -1;
                    int newValue = delta + smoothSpinner.getValue();
                    if (newValue<0) {
                        newValue = 0;
                    }
                    smoothSpinner.getValueFactory().setValue(newValue);
                    lastScoresLength = 0;
                    build();
                }
                    break;
            }
            lineChart.requestLayout();
        }
        private class ScoresSpinnerValueFactory extends SpinnerValueFactory<Integer> {
            public ScoresSpinnerValueFactory() {
                super();
            }
            @Override
            public void decrement(int steps) {
                int value = smoothSpinner.getValue();
                value = Math.max(0, value-steps);
                smoothSpinner.getValueFactory().setValue(value);
            }

            @Override
            public void increment(int steps) {
                int value = smoothSpinner.getValue();
                value = Math.min(scores.size()/2, value+steps);
                smoothSpinner.getValueFactory().setValue(value);
            }

        }
        public ScoresStage() {
            super(StageStyle.DECORATED);
            setTitle("Scores by time. Arrow keys zoom in and out.  s/u smooth/unsmooth. Shift increases adjustment.");
            scene = new Scene(root, width, height, false);
            //imageView.setCache(true);

            smoothSpinner = new Spinner<Integer>(new ScoresSpinnerValueFactory());
            smoothSpinner.getValueFactory().setValue(0);
            System.out.println("Created smoothLabel");
            smoothSpinner.setTranslateX(width-100);
            smoothSpinner.setTranslateY(20);
            smoothSpinner.setMaxWidth(80);

            smoothSpinner.setOnKeyPressed( keyEvent -> { keyEvent.consume(); handleKeyEvent(keyEvent);  });
            smoothSpinner.setOnKeyTyped( keyEvent -> { keyEvent.consume();   });
            smoothSpinner.setFocusTraversable(false);
            smoothLabel.setLabelFor(smoothSpinner);
            smoothLabel.setTranslateX(width-100-65);
            smoothLabel.setTranslateY(23);
            Tooltip tooltip = new Tooltip("Smoothing. Zero means no smoothing");
            Tooltip.install(smoothLabel, tooltip);
            Tooltip.install(smoothSpinner, tooltip);
            xAxis.setLabel("Time");
            yAxis.setLabel("Score");
            lineChart = new LineChart<Number,Number>(xAxis,yAxis);
            lineChart.setLegendVisible(false);
          //  lineChart.setTitle("Scores by time");
            lineChart.getData().add(series);
            lineChart.setMaxWidth(width);
            lineChart.setMaxHeight(height);
            lineChart.setPrefWidth(width);
            lineChart.setPrefHeight(height);
            root.getChildren().addAll(lineChart, smoothLabel, smoothSpinner);
            xAxis.setAutoRanging(true);
            yAxis.setAutoRanging(true);
            yAxis.setForceZeroInRange(false);
            scene.setOnKeyPressed( keyEvent -> {
                try {
                    handleKeyEvent(keyEvent);
                } catch (Throwable thr) {
                    thr.printStackTrace();
                }
            });
            setScene(scene);
            show();
        }
        private List<double[]> smooth() {
            final int smooth = smoothSpinner.getValue();
            final List<double[]> smoothed= new ArrayList<>(scores.size());
            double sum = 0.0;
            final int n=scores.size();
            for(int i=0;i<=smooth;i++) {
                sum+= scores.get(i)[1];
            }
            int count=smooth+1;
            smoothed.add(new double[] {scores.get(0)[0], sum/count});
            for(int i=1;i<n;i++) {
                int drop = i - smooth - 1;
                if (drop >= 0) {
                    sum -= scores.get(drop)[1];
                } else {
                    count++;
                }
                if (i + smooth < n) {
                    sum += scores.get(i + smooth)[1];
                } else {
                    count--;
                }
                smoothed.add(new double[] {scores.get(i)[0],(sum/count)});
            }
            return smoothed;
        }
        private synchronized void build() {
            if (scores.size()<2) {
                return;
            }
            Platform.runLater(new Runnable() {
                @Override
                public void run() {
                    final int smooth = smoothSpinner.getValue();
                    List<double[]> scoresLocal = smooth == 0 ? scores : smooth();
                    if (smooth>0 || lastScoresLength == 0) {
                        lastScoresLength = 0;
                        series.getData().clear();
                    }
                    for(int i=lastScoresLength;i<scoresLocal.size();i++) {
                        double[] pair = scoresLocal.get(i);
                        double time=pair[0];
                        double score =pair[1];
                        maxScore = Math.max(maxScore, score);
                        minScore = Math.min(minScore, score);
                        series.getData().add(new XYChart.Data(time,score));
                    }
                    lastScoresLength = scoresLocal.size();
                    root.requestLayout();
                }});
           // System.out.println("Scores length = " + scores.size());
        }
    }
    private static class MyLabel extends Label {
        public MyLabel(int x,int y) {
            super();
            setTranslateX(x);
            setTranslateY(y);
            Font font = Font.font("monospaced", FontWeight.BOLD, 12.0);
            setFont(font);
        }
        public MyLabel(int x, int y, String text) {
            super();
            setTranslateX(x);
            setTranslateY(y);
            Font font = Font.font("verdana", FontWeight.BOLD, 12.0);
            setTextFill(Color.BLUE);
            setText(text);
            setFont(font);
        }
    }

    private double square (double d) {return d*d;}
    private void calculateMeanAndStd(double [] meanAndStd, INDArray arr) {
        double sum=0.0;
        final long n=arr.length();
        for(long i=0;i<n;i++) {
            double d = arr.getDouble(i);
            sum+=d;
        }
        final double mean = meanAndStd[0]= sum/ n;
        double sumOfSquaredDifferences=0.0;
        for(long i=0;i<n;i++) {
            double d = arr.getDouble(i);
            sumOfSquaredDifferences += square(d-mean);
        }
        meanAndStd[1] = Math.sqrt(sumOfSquaredDifferences/n);
    }
    public void setStatsStorage(StatsStorage statsStorage) {
        this.statsStorage = statsStorage;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
    }

    @Override
    public void onEpochStart(final Model model) {
        System.out.println("Entering onEpochStart");
        Platform.runLater(new Runnable() {
            @Override
            public void run() {
                numberOfParametersLabel.setText(""+model.numParams());
                learningRateLabel.setText(numberFormat7.format(((MultiLayerNetwork) model).getLearningRate(1)));
            }
        });
    }

    @Override
    public void onEpochEnd(Model model) {
    }

    public void setLearningRate(double rate) {
        UILinkedLearningSchedule.learningRate = rate;
    }
    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        final MultiLayerNetwork net = (MultiLayerNetwork) model;
        countOfForwardPasses++;
        final long now = System.currentTimeMillis();

        if (now-lastUpdateMlsForScore> 1000) {
            lastUpdateMlsForScore = now;
            double [] meanAndStd = {0,0};
            final StringBuilder sb = new StringBuilder();
            for(int layer=0;layer<activations.size();layer++) {
                INDArray activation = activations.get(layer);
                calculateMeanAndStd(meanAndStd, activation);
                String message = layer + ": mean = " + numberFormatForMinus.format(meanAndStd[0]) + ", std = " + numberFormatForMinus.format(meanAndStd[1]);
                sb.append(message);
                sb.append("\n");
            }
            Platform.runLater(new Runnable() {
                @Override
                public void run() {
                    activationsText.setText(sb.toString());
                }
            });
        }
    }

    @Override  // Isn't called
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
    }

    private int gradientCalls = 0;
    @Override
    public void onGradientCalculation(Model model) { // This is BEFORE gradients calculation
        UILinkedLearningSchedule.learningRate *= learningRateDecayFactorPerGradientCalculation;
//        if (gradientCalls %1000==0) {
//            System.out.println("In onGradientCalls, learningRate = " + numberFormat7.format(UILinkedLearningSchedule.learningRate));
//        }
        gradientCalls++;
    }

    private static void padRight(StringBuilder sb, String string, int size) {
        sb.append(string);
        int spaces = size-string.length();
        for(int i=0;i<spaces;i++) {
            sb.append(' ');
        }
    }
    private void showUpdatesParametersRatios(StringBuilder sb) {
        for (String sessionId : statsStorage.listSessionIDs()) {
           // System.out.println("SessionId = " + sessionId);
            for (String typeId : statsStorage.listTypeIDsForSession(sessionId)) {
               // System.out.println("  For typeId " + typeId);
                int i = 0;
                for (Persistable persistable : statsStorage.getLatestUpdateAllWorkers(sessionId, typeId)) {
                    //System.out.println(i + ": " + persistable.getClass().getName());
                    if (persistable instanceof SbeStatsReport) {
                        SbeStatsReport sbeStatsReport = (SbeStatsReport) persistable;
                        Map<String, Double> mapUpdates = sbeStatsReport.getMeanMagnitudes(StatsType.Updates); // Activations, Parameters, Gradients, Updates
                        Map<String, Double> mapParameters = sbeStatsReport.getMeanMagnitudes(StatsType.Parameters); // Activations, Parameters, Gradients, Updates
                        for (String key : mapParameters.keySet()) {
                            double update = mapUpdates.get(key);
                            double parameter = mapParameters.get(key);
                            double log10Ratio = Math.log10(update/parameter);
                            padRight(sb, key + ":", 6);
                            sb.append(numberFormat2.format(log10Ratio) + "\n");
                        }
                    }
                }
            }
        }
    }
    private double getVariance(INDArray params) {
        final long n=params.length();
        double sum=0.0;
        for(long i=0;i<n;i++) {
            double v = params.getDouble(i);
            sum += v;
        }
        double mean = sum/n;
        double sumSquared = 0.0;
        for(long i=0;i<n;i++) {
            double v = Math.abs(mean -params.getDouble(i));
            sumSquared+= v*v;
        }
      //  System.out.print("n = " + n + ", mean = " + numberFormat2.format(mean) + ", ");
        return sumSquared/n;
    }
    private void showWeights(MultiLayerNetwork net, StringBuilder sb) {
        for(int i=0;i<net.getnLayers();i++) {
            final Layer layer = net.getLayer(i);
            double varianceParams = getVariance(layer.params());
            double stdParams = Math.sqrt(varianceParams);
            sb.append(i + ": " + numberFormat2.format(stdParams) + "\n");
        }
    }
    private void buildGradientsText(StringBuilder sbGradients, MultiLayerNetwork net) {
        for(int i=0;i<net.getnLayers();i++) {
            final Layer layer = net.getLayer(i);
            INDArray gradients = layer.getGradientsViewArray();
            double[] meanAndStd = {0, 0};
            calculateMeanAndStd(meanAndStd,gradients);
            double meanOfGradients = meanAndStd[0];
            double stdOfGradients = meanAndStd[1];
            sbGradients.append(i + ": mean = " + numberFormatForMinus.format(meanOfGradients)
                + ", std = " + numberFormatForMinus.format(stdOfGradients) + "\n");
        }
    }
    private static String format(double d) {
        return String.format("%7.6e",d);
    }
    private int countScores=0;
    @Override
    public void onBackwardPass(final Model model) { // This is AFTER gradients calculation
        final MultiLayerNetwork net = (MultiLayerNetwork) model;
        final int epoch = net.getEpochCount();
        final long now = System.currentTimeMillis();
        final double runtimeInSeconds = 0.001*(now-startTimeInMillseconds);
        final double score = model.score();

        countScores++;
//        if (countScores%1000==0) {
//            System.out.println(countScores + " scores");
//        }
        boolean added = false;
        if (score<bestScore) {
            bestScore = score;
            bestTime = now;
            added = true;
            double[] timeAndScore = {runtimeInSeconds, score};
            scores.add(timeAndScore);
        }
        if (now-lastUpdateMlsForGradients>3000) {
            if (!added) {
                double[] timeAndScore = {runtimeInSeconds, score};
                scores.add(timeAndScore);
            }
            scoresStage.build();

            lastUpdateMlsForGradients = now;
            final StringBuilder sbGradients = new StringBuilder();
            buildGradientsText(sbGradients, net);
            final StringBuilder weightTextStringBuilder = new StringBuilder();
            showWeights(net, weightTextStringBuilder);
//            INDArray gradients = model.getGradientsViewArray();
//            double[] meanAndStd = {0, 0};
//            calculateMeanAndStd(meanAndStd,gradients);
//            final double stdOfGradients = meanAndStd[1];

            final StringBuilder sbRatios = new StringBuilder();
            if (statsStorage != null) {
                showUpdatesParametersRatios(sbRatios);
            }
            Platform.runLater(new Runnable() {
                @Override
                public void run() {
                    runtimeLabel.setText(numberFormat1.format(runtimeInSeconds));
                    numberOfParametersLabel.setText(""+model.numParams());
                    epochLabel.setText(""+epoch);
                    double secondsSinceBestScore = 0.001*(now-bestTime);
                    String ago= " (" + numberFormat1.format(secondsSinceBestScore) + " secs ago)";
                    bestScoreLabel.setText(numberFormat6.format(bestScore) + ago);
                    scoreLabel.setText(numberFormat6.format(score));
                   // gradientsStdLabel.setText(numberFormat2.format(stdOfGradients));
                    learningRateLabel.setText(format(UILinkedLearningSchedule.learningRate));
                    gradientsStdText.setText(sbGradients.toString());
                    weightsText.setText(weightTextStringBuilder.toString());
                    if (sbRatios.length()>0) {
                        ratioText.setText(sbRatios.toString());
                    }
                }
            });
        }
    }

    private void addKeyListener(Scene scene) {
        scene.setOnKeyPressed(keyEvent -> {
            KeyCode code = keyEvent.getCode();
            switch (code) {
                case DOWN: case UP: {
                    double f = keyEvent.isShiftDown()? 2.0: 1.1;
                    double factor = code.equals(KeyCode.DOWN)? (1.0/f) : f;
                    UILinkedLearningSchedule.learningRate *= factor;
                    learningRateLabel.setText(format(UILinkedLearningSchedule.learningRate));
                }
                break;
                case PAGE_UP: case PAGE_DOWN: {
                    double f = keyEvent.isShiftDown()? 1.000_01: 1.000_001;
                    double factor = code.equals(KeyCode.PAGE_DOWN)? (1.0/f) : f;
                    learningRateDecayFactorPerGradientCalculation = Math.min(0.999_9999, factor*learningRateDecayFactorPerGradientCalculation);
                    learningRateDecayLabel.setText(format(learningRateDecayFactorPerGradientCalculation));
                }
                    break;
            }
        });
    }
    // From https://stackoverflow.com/questions/26854301/how-to-control-the-javafx-tooltips-delay

    private void makeTexts() {
        int y = 30;

        epochLabelLabel = new MyLabel(15,y-20, "Epoch: ");
        epochLabel = new MyLabel(15+65,y-20);
//        iterationLabel = new MyLabel(120,y);
        numberOfParametersLabelLabel = new MyLabel(15,y, "Params: ");
        numberOfParametersLabel = new MyLabel(15+64,y);

        learningRateLabelLabel = new MyLabel(140,y, "LR: ");
        learningRateLabel = new MyLabel(140+30,y);
        learningRateLabelLabel.setTextFill(Color.GREEN);

        learningRateDecayLabelLabel = new MyLabel(140, y-20, "Decay: ");
        learningRateDecayLabel = new MyLabel(50+140, y-20);
        learningRateDecayLabel.setText(format(learningRateDecayFactorPerGradientCalculation));
        learningRateDecayLabelLabel.setTextFill(Color.GREEN);

        bestScoreLabelLabel = new MyLabel(290, y-20, "Best: ");
        bestScoreLabel = new MyLabel(290+50, y-20);

        scoreLabelLabel = new MyLabel(290,y, "Score: ");
        scoreLabel = new MyLabel(50+290,y);

        int yOffset = 26;
        activationsLabel = new MyLabel(15,y+40,"Activations");
        activationsText = new Text(15, y+40+yOffset,"");
        gradientsTextLabel = new MyLabel(250,y+40,"Gradients");
        gradientsStdText = new Text(250, y+40+yOffset,"");

        ratiosLabel = new MyLabel(480,y+40, "Log Ratios");
        ratioText = new Text(480, y+40+yOffset, "");

        weightsLabel = new MyLabel(590, y+40, "Weights std");
        weightsText = new Text(590, y+40+yOffset,"");

        runtimeLabelLabel = new MyLabel(545,10,"Runtime in secs:");
        runtimeLabel = new MyLabel(545,30);

        Tooltip tooltipDecay = new Tooltip("Learning rate decay factor.\nUse Up/Down arrows to change this.");
        tooltipDecay.setShowDelay(Duration.seconds(1));
        Tooltip.install(learningRateDecayLabelLabel, tooltipDecay);
        Tooltip.install(learningRateDecayLabel, tooltipDecay);

        Tooltip tooltipBest = new Tooltip("Best score so far");
        tooltipBest.setShowDelay(Duration.seconds(1));
        Tooltip.install(bestScoreLabelLabel, tooltipBest);
        Tooltip.install(bestScoreLabel, tooltipBest);

        Tooltip tooltipLearningRate = new Tooltip("Learning rate.\nUse Up/Down arrows to change this.");
        tooltipLearningRate.setShowDelay(Duration.seconds(1));
        Tooltip.install(learningRateLabelLabel, tooltipLearningRate);
        Tooltip.install(learningRateLabel, tooltipLearningRate);

        Tooltip tooltipRatio = new Tooltip("Log mean ratio of updates to parameters:\nlog10(mean(abs(updates))/mean(abs(parameters))");
        tooltipRatio.setShowDelay(Duration.seconds(1));
        Tooltip.install(ratioText, tooltipRatio);
        Tooltip.install(ratiosLabel, tooltipRatio);
        Tooltip tooltipWeights = new Tooltip("Standard deviation of weights per layer");
        tooltipWeights.setShowDelay(Duration.seconds(1));
        Tooltip.install(weightsLabel, tooltipWeights);
        Tooltip.install(weightsText, tooltipWeights);

        Tooltip toolTipGradients = new Tooltip("Gradients per layer");
        toolTipGradients.setShowDelay(Duration.seconds(1));
        Tooltip.install(gradientsTextLabel, toolTipGradients);
        Tooltip.install(gradientsStdText, toolTipGradients);

        Tooltip tooltipActivations = new Tooltip("Activations per layer");
        tooltipActivations.setShowDelay(Duration.seconds(1));
        Tooltip.install(activationsLabel, tooltipActivations);
        Tooltip.install(activationsText, tooltipActivations);

        Font textFont = Font.font("monospaced", FontWeight.BOLD, 12.0);
        activationsText.setFont(textFont);
        gradientsStdText.setFont(textFont);
        ratioText.setFont(textFont);
        weightsText.setFont(textFont);

        root.getChildren().addAll(epochLabel, numberOfParametersLabelLabel, numberOfParametersLabel,
            bestScoreLabel, scoreLabelLabel,scoreLabel,
            epochLabelLabel,learningRateLabelLabel,bestScoreLabelLabel, gradientsTextLabel,
            activationsLabel, ratiosLabel,weightsLabel,
            learningRateDecayLabelLabel,learningRateDecayLabel, learningRateLabel, activationsText, gradientsStdText, ratioText, weightsText,
            runtimeLabel,runtimeLabelLabel
        );
    }
    @Override
    public void start(Stage stage) throws Exception {
        stage.setTitle("Up/Down arrows adjust learning rate; Page Up/Down adjust decay factor. Shift increases adjustment.");
        root = new Group();
        Scene scene = new Scene(root, WIDTH, HEIGHT, false);
        makeTexts();
        scene.setFill(new Color(0.9, 0.9, 0.95, 1));
        stage.setScene(scene);
        stage.setFullScreen(false);
        stage.toFront();
        addKeyListener(scene);
        scoresStage = new ScoresStage();
        stage.show();
    }

    public static TrainingListenerWithUI initialize(final double learningRate, final StatsStorage statsStorage) {
        final Runnable runnable = new Runnable() {
            @Override
            public void run() {
                launch(TrainingListenerWithUI.class,new String[]{});
            }
        };
        new Thread(runnable).start();

        while (TrainingListenerWithUI.instance == null) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException exc) {
            }
            try {Thread.sleep(50);} catch (InterruptedException exc) {}
        }
        try {Thread.sleep(500);} catch (InterruptedException exc) {}
        instance.setLearningRate(learningRate);
        instance.setStatsStorage(statsStorage);
        return instance;
    }
}
