from multiscale import MultiscaleFeaturesExtractor

# Import functions from scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split

import matplotlib.pyplot as plt
import numpy as np
import time


def train_model(model, training_features, training_labels,
                eval_features, eval_labels, batch_size, n_epochs):
    
    N = training_features.shape[0]

    n_step = int(batch_size*N/n_epochs)
    training_loss = np.zeros(n_step)
    eval_loss = np.zeros(n_step)

    for i in range(n_step):
        batch_index = np.random.randint(0, N, batch_size)
        model.fit(training_features[batch_index], training_labels[batch_index])
        
        train_pred = model.predict(training_features)
        eval_pred = model.predict(eval_features)

        training_loss[i] = accuracy_score(training_labels, train_pred)
        eval_loss[i] = accuracy_score(eval_labels, eval_pred)
    
    return model, training_loss, eval_loss





if __name__ == '__main__':

    train = False
    if train:

        training_path = '../data/Benchmark_MVA/training_lille1'
        eval_path = '../data/Benchmark_MVA/eval_paris'


        print('Collect Training Features')
        t0 = time.time()

        # Create a feature extractor
        f_extractor = MultiscaleFeaturesExtractor(training_path, [0.5, 1, 5])

        # Collect training features and labels
        training_features, training_labels = f_extractor.extract_training()

        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))


        train = False
        print('Training Random Forest')
        t0 = time.time()

        clf = RandomForestClassifier()        
        # Perform cross validation and show accuracy per category
        predicted_labels = cross_val_predict(clf, training_features, training_labels)
        print('\nOverall Classification Report:\n', classification_report(training_labels, predicted_labels))

        # Create and train a random forest with scikit-learn
        clf = RandomForestClassifier()


        clf.fit(training_features, training_labels)


        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))



    # Print loss
    train2 = False
    if train2:

        training_path = '../data/Benchmark_MVA/training_lille1'
        eval_path = '../data/Benchmark_MVA/eval_paris'

        print('Collect Training Features')
        t0 = time.time()

        # Create a feature extractor
        extractor_train = MultiscaleFeaturesExtractor(training_path, [0.5, 1, 5])

        extractor_eval = MultiscaleFeaturesExtractor(eval_path, [0.5, 1, 5])

        # Collect features and labels
        training_features, training_labels = extractor_train.extract_training()
        eval_features, eval_labels = extractor_eval.extract_training()

        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

        print('Training Random Forest')
        t0 = time.time()

        clf = RandomForestClassifier()   
        _, train_loss, eval_loss = train_model(clf, training_features, training_labels,
                                               eval_features, eval_labels, n_epochs=10,
                                               batch_size=50)
        
        fig, ax = plt.subplots()

        ax.plot(train_loss, color='green', label='Train loss')
        ax.plot(eval_loss, linestyle='dashed', color='red', label='Eval loss')
        fig.legend()
        plt.show()
        fig.savefig('../results/learning_curve.png')



    if True:
        print(train_loss)

    # Test
    # ****
    #
    test = False
    if test:
        print('Compute testing features')
        t0 = time.time()

        # Collect test features
        test_features = f_extractor.extract_test(test_path)

        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

        print('Test')
        t0 = time.time()

        # Test the random forest on our features
        predictions = clf.predict(test_features)

        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

        # Save prediction for submission
        # ******************************
        #

        print('Save predictions')
        t0 = time.time()
        np.savetxt('MiniDijon9.txt', predictions, fmt='%d')
        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))