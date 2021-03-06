package com.helencoder.nlp.text;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

/**
 * ParagraphVectors
 *
 * Created by helencoder on 2017/12/14.
 */
public class ParagraphVectorsTextExample {

    private static Logger log = LoggerFactory.getLogger(ParagraphVectorsTextExample.class);

    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

        // 方式二、读取文件

        AbstractCache<VocabWord> cache = new AbstractCache<>();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        /*
             if you don't have LabelAwareIterator handy, you can use synchronized labels generator
              it will be used to label each document/sequence/line with it's own label.

              But if you have LabelAwareIterator ready, you can can provide it, for your in-house labels
        */
        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(5)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(false)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

        vec.fit();

        for (String label : source.getLabels()) {
            System.out.println(label);
        }

        vec.extractLabels();

        System.exit(0);

        /*
            In training corpus we have few lines that contain pretty close words invloved.
            These sentences should be pretty close to each other in vector space

            line 3721: This is my way .
            line 6348: This is my case .
            line 9836: This is my house .
            line 12493: This is my world .
            line 16393: This is my work .

            this is special sentence, that has nothing common with previous sentences
            line 9853: We now have one .

            Note that docs are indexed from 0
         */

        Collection<String> lst = vec.nearestLabels("This is my houses .", 10);
        System.out.println("10 Labels closest to 'This is my houses .': " + lst);

        double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
        log.info("9836/12493 ('This is my house .'/'This is my world .') similarity: " + similarity1);

        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
        log.info("3721/16393 ('This is my way .'/'This is my work .') similarity: " + similarity2);

        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
        log.info("6348/3721 ('This is my case .'/'This is my way .') similarity: " + similarity3);

        // likelihood in this case should be significantly lower
        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
        log.info("3721/9853 ('This is my way .'/'We now have one .') similarity: " + similarityX +
                "(should be significantly lower)");
    }
}
