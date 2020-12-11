/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.quickstart.modeling.feedforward.classification;


import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.wip.javafxui.TrainingListenerWithUI;
import org.deeplearning4j.examples.wip.javafxui.UILinkedLearningSchedule;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.Map;


/** A slightly more involved multilayered (MLP) applied to digit classification for the MNIST dataset (http://yann.lecun.com/exdb/mnist/).
*
* This example uses two input layers and one hidden layer.
*
* The first input layer has input dimension of numRows*numColumns where these variables indicate the
* number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
* (relu) activation function. The weights for this layer are initialized by using Xavier initialization
* (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
* to avoid having a steep learning curve. This layer sends 500 output signals to the second layer.
*
* The second input layer has input dimension of 500. This layer also uses a rectified linear unit
* (relu) activation function. The weights for this layer are also initialized by using Xavier initialization
* (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
* to avoid having a steep learning curve. This layer sends 100 output signals to the hidden layer.
*
* The hidden layer has input dimensions of 100. These are fed from the second input layer. The weights
* for this layer is also initialized using Xavier initialization. The activation function for this
* layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
* add up to 1. The highest of these normalized values is picked as the predicted class.
*
*/
public class MNISTDoubleLayerWithUI {

    private static Logger log = LoggerFactory.getLogger(MNISTDoubleLayerWithUI.class);

    private static void populateBigNetFromSmallerNet(MultiLayerNetwork small, MultiLayerNetwork big) {
        Map<String, INDArray> bigParams = big.paramTable();
        Map<String, INDArray> smallParams = small.paramTable();
        if (!bigParams.keySet().equals(smallParams.keySet())) {
            System.err.println(smallParams.keySet());
            System.err.println(bigParams.keySet());
            throw new IllegalArgumentException("Can't populate big from small, they have different parameters");
        }
        for(String param: smallParams.keySet()) {
            INDArray smallIndArray = smallParams.get(param);
            INDArray bigIndArray = bigParams.get(param);
            long []smallShape = smallIndArray.shape();
            long []bigShape = bigIndArray.shape();
            System.out.println(param + ": " + Arrays.toString(smallShape) + ", " + Arrays.toString(bigShape));
            if (smallShape.length!=bigShape.length) {
                throw new IllegalArgumentException("Different shapes for param " + param + ": " +
                    Arrays.toString(smallShape) + ", " + Arrays.toString(bigShape));
            }
            for(int i=0;i<smallShape.length;i++) {
                if (smallShape[i]>bigShape[i]) {
                    throw new IllegalArgumentException("Wrong order of small and big");
                }
            }
            if (smallShape.length==2) {
                for(long i=0;i<smallShape[0];i++) {
                    for(long j=0;j<smallShape[1];j++) {
                        bigIndArray.putScalar(i,j,smallIndArray.getFloat(i,j));
                    }
                }
            } else {
                throw new IllegalArgumentException("We only handle 2d shapes, got " + smallShape.length);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 15; // number of epochs to perform
        double rate = 0.0015; // learning rate

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        DataSet dataset = mnistTrain.next();
        // For MNist:
        //System.out.println("data set features: " + dataset.getFeatures().shapeInfoToString());
        //System.out.println("data set labels: " + dataset.getLabels().shapeInfoToString());
        //System.exit(0);
        //data set features: Rank: 2, DataType: FLOAT, Offset: 0, Order: c, Shape: [64,784],  Stride: [784,1]
        //data set labels: Rank: 2, DataType: FLOAT, Offset: 0, Order: c, Shape: [64,10],  Stride: [10,1]

        long startTime= System.currentTimeMillis();
        int layer1Size=500;
        int layer2Size=100;

        final StatsStorage statsStorage = new InMemoryStatsStorage(); // das
        TrainingListenerWithUI trainingListenerWithUI = TrainingListenerWithUI.initialize(rate, statsStorage); // das
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nadam(UILinkedLearningSchedule.getInstance(rate)))
            .l2(rate * 0.005) // regularize learning model
            .list()
            .layer(new DenseLayer.Builder() //create the first input layer.
                    .nIn(numRows * numColumns)
                    .nOut(layer1Size)
                    .build())
            .layer(new DenseLayer.Builder() //create the second input layer
                    .nIn(layer1Size)
                    .nOut(layer2Size)
                    .build())
            .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .activation(Activation.SOFTMAX)
                    .nOut(outputNum)
                    .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf); model.init();

      //  MultiLayerNetwork modelSmall = MultiLayerNetwork.load(new File("d:/tmp/mnist.50-40.saved"),true);
      //  populateBigNetFromSmallerNet(modelSmall,model);
        model.setListeners(trainingListenerWithUI, new StatsListener(statsStorage));  //print the score with every iteration

        log.info("Train model....");
        model.fit(mnistTrain, numEpochs);

        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(mnistTest);

        log.info(eval.stats());
        double seconds = 0.001*(System.currentTimeMillis()-startTime);
        log.info("****************Example finished********************, seconds = " + seconds);
        model.save(new File("d:/tmp/mnist."+ layer1Size + "-"+ layer2Size + ".saved"),true);

    }

}
