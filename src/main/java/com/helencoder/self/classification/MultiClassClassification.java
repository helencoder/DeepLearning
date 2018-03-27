package com.helencoder.self.classification;

import com.helencoder.self.Doc2Vec;
import com.helencoder.self.util.NewsIterator;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 多类别分类
 *
 * Created by helencoder on 2018/2/23.
 */
public class MultiClassClassification {
    public static void main(String[] args) throws Exception {
        /**
         * 实现方案:
         *  1、word2vec + lstm
         *  2、doc2vec +lstm
         *  3、LPA
         */

        // 模型训练样本文件
        String classPathResource = new ClassPathResource("DengTaData").getFile().getAbsolutePath() + File.separator;
        String filePath = new File(classPathResource + File.separator + "data.txt").getAbsolutePath();

        // 类别训练文件
        String classFilePath = classPathResource + "LabelledNews";

        // 方案验证
        //word2vecLstm(filePath);

        doc2vecLstm("docLstm.model");

    }

    /**
     * word2vec + lstm
     */
    public static void word2vecLstm(String filePath) throws Exception {
        // 训练word2vec模型
        String modelPath = "dengta.model";
        //Word2VecModel.train(filePath, modelPath);

        // 进行模型构建
        String userDirectory = new ClassPathResource("DengTaData").getFile().getAbsolutePath() + File.separator;
        String DATA_PATH = userDirectory + "LabelledNews";

        int batchSize = 50;     //Number of examples in each minibatch
        int nEpochs = 1000;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 300;  //Truncate reviews with length (# words) greater than this

        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(new File(modelPath));

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        NewsIterator iTrain = new NewsIterator.Builder()
                .dataDirectory(DATA_PATH)
                .wordVectors(wordVectors)
                .batchSize(batchSize)
                .truncateLength(truncateReviewsToLength)
                .tokenizerFactory(tokenizerFactory)
                .train(true)
                .build();

        NewsIterator iTest = new NewsIterator.Builder()
                .dataDirectory(DATA_PATH)
                .wordVectors(wordVectors)
                .batchSize(batchSize)
                .tokenizerFactory(tokenizerFactory)
                .truncateLength(truncateReviewsToLength)
                .train(false)
                .build();

        //DataSetIterator train = new AsyncDataSetIterator(iTrain,1);
        //DataSetIterator test = new AsyncDataSetIterator(iTest,1);

        int inputNeurons = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length; // 100 in our case
        int outputs = iTrain.getLabels().size();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(0.0018)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(inputNeurons).nOut(200)
                        .activation(Activation.SOFTSIGN).build())
                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(outputs).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        System.out.println("Starting training");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(iTrain);
            iTrain.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = net.evaluate(iTest);

            System.out.println(evaluation.stats());
        }

        ModelSerializer.writeModel(net, userDirectory + "NewsModel.net", true);
        System.out.println("----- Example complete -----");

    }

    /**
     * doc2vec classification
     */
    public static void doc2vecLstm(String modelPath) throws Exception {
        // 训练doc2vec模型
        String vectorPath = "dengtaDoc.model";
        //Doc2Vec.train(filePath, modelPath);

        // 进行模型构建
        String userDirectory = new ClassPathResource("DengTaData").getFile().getAbsolutePath() + File.separator;
        String DATA_PATH = userDirectory + "LabelledNews";
        String dataPath = userDirectory + "data.txt";

        // 加载训练好的Doc2Vec模型
        ParagraphVectors vectors = Doc2Vec.load(vectorPath);

        // Doc2Vec模型对应关系
        List<String> fileList = getFileDataByLine(dataPath);
        Map<String, double[]> fileVecMap = new HashMap<>();
        int count = 0;
        for (String file : fileList) {
            String tag = "DOC_" + count;
            fileVecMap.put(file, vectors.getWordVector(tag));
            count++;
        }



    }

    /**
     * LPA
     */
    public static void lpa() {

    }


    // get the file data by line
    public static List<String> getFileDataByLine(String filepath) {
        List<String> fileDataList = new ArrayList<String>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filepath)));
            for (String line = br.readLine(); line != null; line = br.readLine()) {
                // handle
                if (line.length() != 0) {
                    fileDataList.add(line);
                }
            }
            br.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return fileDataList;
    }

}
