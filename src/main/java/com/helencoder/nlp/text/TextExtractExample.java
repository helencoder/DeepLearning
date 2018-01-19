package com.helencoder.nlp.text;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * 基于DeepLearning的文本提取示例
 *
 * Created by helencoder on 2017/12/14.
 */
public class TextExtractExample {

    private static Logger log = LoggerFactory.getLogger(TextExtractExample.class);

    public static void main(String[] args) throws Exception {
        String filePath = new ClassPathResource("NewsData/article.txt").getFile().getAbsolutePath();
        log.info("Load & Vectorize Articles....");

        SentenceIterator iter = new BasicLineIterator(filePath);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        VocabCache<VocabWord> cache = new AbstractCache<>();
        WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>()
                .vectorLength(200)
                .useAdaGrad(false)
                .cache(cache).build();

        log.info("Building Word2Vec model....");

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .epochs(1)
                .layerSize(100)
                .seed(42)
                .windowSize(3)
                .iterate(iter)
                .tokenizerFactory(t)
                .lookupTable(table)
                .vocabCache(cache)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("市场", 10);
        System.out.println("10 Words closest to 'day': " + lst);


        log.info("Writing word vectors to text file....");
        WordVectorSerializer.writeWord2VecModel(vec, "src/main/resources/NewsData/article.model");

        log.info("Word2vec uptraining...");

        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("src/main/resources/NewsData/article.model");

        SentenceIterator iterator = new BasicLineIterator("src/main/resources/NewsData/article_new.txt");
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        word2Vec.setTokenizerFactory(tokenizerFactory);
        word2Vec.setSentenceIterator(iterator);


        log.info("Word2vec uptraining...");

        word2Vec.fit();

        lst = word2Vec.wordsNearest("市场", 10);
        log.info("Closest words to 'day' on 2nd run: " + lst);

    }
}
