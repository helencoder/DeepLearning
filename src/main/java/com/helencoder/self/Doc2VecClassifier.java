package com.helencoder.self;

import com.helencoder.nlp.paragraphvectors.util.LabelSeeker;
import com.helencoder.nlp.paragraphvectors.util.MeansBuilder;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * 基于Doc2Vec的分类器
 *
 * Created by helencoder on 2018/2/27.
 */
public class Doc2VecClassifier {
    public static void main(String[] args) throws Exception {
        /**
         * 实现思路
         * 1、文本组织: 一类文本为一个文件,一行一篇文本
         */

        run();
    }

    public static void run() throws Exception {
        // 构建doc2Vec模型
        ClassPathResource resource = new ClassPathResource("paravec/labeled");

        // build a iterator for our dataset
        LabelAwareIterator iterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(resource.getFile())
                .build();

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();

        // Start model training
        paragraphVectors.fit();

        // 样本预测
        ClassPathResource unClassifiedResource = new ClassPathResource("paravec/unlabeled");
        FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(unClassifiedResource.getFile())
                .build();

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        while (unClassifiedIterator.hasNextDocument()) {
            LabelledDocument document = unClassifiedIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            System.out.println("Document '" + document.getLabels() + "' falls into the following categories: ");
            for (Pair<String, Double> score: scores) {
                System.out.println("        " + score.getFirst() + ": " + score.getSecond());
            }
        }


    }


}
