import os
import pickle
import sys
from optparse import OptionParser
from time import time

import numpy as np
import tensorflow as tf
from joblib import load

######### Tensorflow settings ##########
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

############ Append sys path ###########
sys.path.append('../')

from modules.statistic_processor import (compute_class_mean,
                                         get_classification_report,
                                         get_complete_evaluation,
                                         get_misclassifications)
from modules.plot_processor import (plot_class_means, plot_class_overlay,
                                    plot_heatmap, plot_patch_and_dist,
                                    plot_series_and_dist)
from modules.patch_generator import (compute_trivial_preds,
                                     create_histo_dataset, get_all_patch,
                                     get_all_patch_params,
                                     get_data_patch_stats,
                                     get_generator_id_list,
                                     get_patch_params_list, get_sample_id_list)
from modules.model_trainer import (predict_clf, train_blackbox, train_clf,
                                   train_descriptive)
from modules.model import create_clf, create_dense_model, create_model
from modules.file_writer import write_to_file
from modules.feature_extractor import (compute_relevant_subset_features,
                                       create_clf_features)
from modules.data_processor import generate_data
from modules.data_generator import DataGenerator, DataGenerator_sample

########################################
############ Not modulazied ############
########################################


def validate_and_adjust_settings(zero, attach, notemp):
    # 0 0 0 invalid
    # 1 0 0 valid
    # 0 1 0 valid
    # 1 1 0 valid
    # 0 0 1 invalid
    # 1 0 1 valid
    # 0 1 1 invalid
    # 1 1 1 valid
    if zero == 0 and attach == 0:
        return 1, attach, notemp
    if attach == 0 and notemp == 0:
        return 1, attach, notemp
    if notemp == 1:
        return 1, attach, notemp
    return zero, attach, notemp


def define_setup(config, zero, attach, notemp):
    s = 'strides_'
    l = 'length_'
    for c in config:
        s += str(c[0]) + '-'
        l += str(c[1]) + '-'
    s = s[:-1] + '_' + l[:-1] + '_zero-'
    s += '1' if zero else '0'
    s += '_attach-'
    s += '1' if attach else '0'
    s += '_notemp-'
    s += '1' if notemp else '0'
    return s

########################################
############## Main ####################
########################################


def process(options):
    # adjust parameter
    strides = np.fromstring(options.strides, dtype=int, sep=',')
    length = np.fromstring(options.length, dtype=int, sep=',')
    config = []
    for i in range(len(strides)):
        config.append([strides[i], length[i]])

    sample_mean_classes = np.fromstring(
        options.mean_classes, dtype=int, sep=',')

    ########### Prepare Data ###############
    trainX, trainY, valX, valY, testX, testY, classes, seqlen, channel = generate_data(
        options.path, create_val=True, verbose=1)
    trainLen, valLen, testLen = trainX.shape[0], valX.shape[0], testX.shape[0]

    set_name = options.path.split(os.sep)[-2]
    model_path = os.path.join('../models', set_name)
    img_path = os.path.join('../images', set_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    ############# Store the times ##########
    times = {}

    ############# Patch Data ###############
    if options.include_l1 or options.include_l2:
        # [Stride, Length]
        options.zero, options.attach, options.notemp = validate_and_adjust_settings(
            options.zero, options.attach, options.notemp)

        params = {'dim': [seqlen, channel], 'batch_size': 32, 'config': config,
                  'zero': options.zero, 'attach': options.attach, 'notemp': options.notemp, 'shuffle': False}

        setup = define_setup(config, options.zero,
                             options.attach, options.notemp)
        setup_path = os.path.join(model_path, setup)
        if not os.path.exists(setup_path):
            os.makedirs(setup_path)
        image_path = os.path.join(img_path, setup, options.clf_type)
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # Generators
        s_time = time()
        trainIds = get_generator_id_list(trainLen, seqlen, config)
        train_generator = DataGenerator(trainIds, trainX, trainY, **params)
        times['Train Generator L1'] = time() - s_time
        s_time = time()
        valIds = get_generator_id_list(valLen, seqlen, config)
        val_generator = DataGenerator(valIds, valX, valY, **params)
        times['Val Generator L1'] = time() - s_time
        s_time = time()
        testIds = get_generator_id_list(testLen, seqlen, config)
        test_generator = DataGenerator(testIds, testX, testY, **params)
        times['Test Generator L1'] = time() - s_time

    ############ Train Level 1 #############
    if options.include_l1 or options.include_trivial:
        input_shape = trainX.shape[1:]
        if options.attach:
            input_shape = list(input_shape)
            input_shape[-1] += 1
            input_shape = tuple(input_shape)
        patch_model_path = os.path.join(setup_path, 'patch_classifier.h5')

        if os.path.exists(patch_model_path) and options.load_l1:
            patch_model = tf.keras.models.load_model(patch_model_path)
        else:
            patch_model = create_model(input_shape, classes)
            s_time = time()
            patch_model = train_descriptive(patch_model_path, patch_model, trainIds, valIds,
                                            trainX, trainY, valX, valY, params, thresh=0.0, verbose=1, workers=1)
            times['Training L1'] = time() - s_time

        s_time = time()
        softmax_trainXp = patch_model.predict(train_generator)[:len(trainIds)]
        times['Train Prediction L1'] = time() - s_time
        s_time = time()
        softmax_valXp = patch_model.predict(val_generator)[:len(valIds)]
        times['Val Prediction L1'] = time() - s_time
        s_time = time()
        softmax_testXp = patch_model.predict(test_generator)[:len(testIds)]
        times['Test Prediction L1'] = time() - s_time

    ############# Prepare Level ############
    if options.include_l2 or options.include_trivial:
        s_time = time()
        train_pps = get_data_patch_stats(trainLen, seqlen, config)[1]
        train_sidx = get_sample_id_list(trainLen, train_pps)
        times['Train Extraction L1'] = time() - s_time
        s_time = time()
        val_pps = get_data_patch_stats(valLen, seqlen, config)[1]
        val_sidx = get_sample_id_list(valLen, val_pps)
        times['Val Extraction L1'] = time() - s_time
        s_time = time()
        test_pps = get_data_patch_stats(testLen, seqlen, config)[1]
        test_sidx = get_sample_id_list(testLen, test_pps)
        times['Test Extraction L1'] = time() - s_time

    ########### Trivial Classifier #########
    if options.include_trivial:
        s_time = time()
        trivial_train_preds = compute_trivial_preds(
            softmax_trainXp, train_sidx, mode=options.trivial_mode)
        times['Train Prediction Trivial'] = time() - s_time
        s_time = time()
        trivial_val_preds = compute_trivial_preds(
            softmax_valXp, val_sidx, mode=options.trivial_mode)
        times['Val Prediction Trivial'] = time() - s_time
        s_time = time()
        trivial_test_preds = compute_trivial_preds(
            softmax_testXp, test_sidx, mode=options.trivial_mode)
        times['Test Prediction Trivial'] = time() - s_time
        s_time = time()

    ############ Train Level 2 #############
    if options.include_l2:
        s_time = time()
        histo_trainX = create_histo_dataset(softmax_trainXp, train_sidx)
        times['Train Extraction L2'] = time() - s_time
        s_time = time()
        histo_valX = create_histo_dataset(softmax_valXp, val_sidx)
        times['Val Extraction L2'] = time() - s_time
        s_time = time()
        histo_testX = create_histo_dataset(softmax_testXp, test_sidx)
        times['Test Extraction L2'] = time() - s_time

        clf_model_path = os.path.join(
            setup_path, options.clf_type + '_classifier.pickle')
        if os.path.exists(clf_model_path) and options.load_l2:
            clf = load(clf_model_path)
        else:
            clf = create_clf(options.clf_type)
            s_time = time()
            clf = train_clf(clf_model_path, clf, histo_trainX,
                            trainY, histo_valX, valY)
            times['Training L2'] = time() - s_time

        s_time = time()
        clf_train_pred = clf.predict(histo_trainX)
        times['Train Prediction L2'] = time() - s_time
        s_time = time()
        clf_val_pred = clf.predict(histo_valX)
        times['Val Prediction L2'] = time() - s_time
        s_time = time()
        clf_test_pred = clf.predict(histo_testX)
        times['Test Prediction L2'] = time() - s_time

    ############ Train Blackbox ############
    if options.include_blackbox:
        # Generators
        params_simple = {'dim': [seqlen, channel],
                         'batch_size': 32, 'shuffle': False}
        s_time = time()
        train_generator_simple = DataGenerator_sample(
            np.arange(trainLen), trainX, trainY, **params_simple)
        times['Train Generator Blackbox'] = time() - s_time
        s_time = time()
        val_generator_simple = DataGenerator_sample(
            np.arange(valLen), valX, valY, **params_simple)
        times['Val Generator Blackbox'] = time() - s_time
        s_time = time()
        test_generator_simple = DataGenerator_sample(
            np.arange(testLen), testX, testY, **params_simple)
        times['Test Generator Blackbox'] = time() - s_time

        blackbox_model_path = os.path.join(
            model_path, 'blackbox_classifier.h5')
        if os.path.exists(blackbox_model_path) and options.load_blackbox:
            blackbox_model = tf.keras.models.load_model(blackbox_model_path)
        else:
            blackbox_model = create_model(trainX.shape[1:], classes)
            s_time = time()
            blackbox_model = train_blackbox(
                blackbox_model_path, blackbox_model, train_generator_simple, val_generator_simple, epochs=50, verbose=1, workers=1)
            times['Training Blackbox'] = time() - s_time

        s_time = time()
        bm_train_pred = np.argmax(blackbox_model.predict(
            train_generator_simple), axis=-1)[:trainLen]
        times['Train Prediction Blackbox'] = time() - s_time
        s_time = time()
        bm_val_pred = np.argmax(blackbox_model.predict(
            val_generator_simple), axis=-1)[:valLen]
        times['Val Prediction Blackbox'] = time() - s_time
        s_time = time()
        bm_test_pred = np.argmax(blackbox_model.predict(
            test_generator_simple), axis=-1)[:testLen]
        times['Test Prediction Blackbox'] = time() - s_time

    ############ Train SimpleClf ##########
    if options.include_simple_clf:
        simple_clf_model_path = os.path.join(
            model_path, 'simpleClf_' + options.clf_type + '_classifier.pickle')
        if os.path.exists(simple_clf_model_path) and options.load_simple_clf:
            simple_clf_model = load(simple_clf_model_path)
        else:
            simple_clf_model = create_clf(options.clf_type)
            s_time = time()
            simple_clf_model = train_clf(simple_clf_model_path, simple_clf_model, trainX,
                                         trainY, valX, valY)
            times['Training SimpleClf_' + options.clf_type] = time() - s_time

        s_time = time()
        simple_clf_train_pred = predict_clf(simple_clf_model, trainX)
        times['Train Prediction SimpleClf_' + options.clf_type] = time() - \
            s_time
        s_time = time()
        simple_clf_val_pred = predict_clf(simple_clf_model, valX)
        times['Val Prediction SimpleClf_' + options.clf_type] = time() - s_time
        s_time = time()
        simple_clf_test_pred = predict_clf(simple_clf_model, testX)
        times['Test Prediction SimpleClf_' + options.clf_type] = time() - \
            s_time

    ############ Compute Features ###########
    if options.include_feature_simple_clf or options.include_feature_blackbox or options.compute_features or options.force_compute_features:
        if options.feature_mode == 'subset':
            feature_path = os.path.join(
                model_path, 'features_' + options.feature_mode + '_' + str(options.feature_subset) + '.pickle')
        else:
            feature_path = os.path.join(
                model_path, 'features_' + options.feature_mode + '.pickle')

        if os.path.exists(feature_path) and not options.force_compute_features:
            with open(feature_path, 'rb') as f:
                train_features, val_features, test_features = pickle.load(f)
        else:
            s_time = time()
            if options.feature_mode == 'subset':
                f_mode = 'given'
                relevant_features = compute_relevant_subset_features(
                    trainX, trainY, classes, num=options.feature_subset)
            else:
                f_mode = 'relevant'
                relevant_features = None

            train_features, relevant_features = create_clf_features(
                trainX, trainY, pre_selected=relevant_features, mode=f_mode, return_names=True)
            times['Train Featueextraction'] = time() - s_time
            s_time = time()
            val_features = create_clf_features(
                valX, pre_selected=relevant_features, mode='given')
            times['Val Featueextraction'] = time() - s_time
            s_time = time()
            test_features = create_clf_features(
                testX, pre_selected=relevant_features, mode='given')
            times['Test Featueextraction'] = time() - s_time

            with open(feature_path, 'wb') as f:
                pickle.dump(
                    [train_features, val_features, test_features], f)

    ####### Train Feature SimpleClf ########
    if options.include_feature_simple_clf:
        simple_clf_model_feature_path = os.path.join(
            model_path, 'simpleClf_' + options.clf_type + '_classifier_feature.pickle')
        if os.path.exists(simple_clf_model_feature_path) and options.load_feature_simple_clf:
            simple_clf_feature_model = load(simple_clf_model_feature_path)
        else:
            simple_clf_feature_model = create_clf(options.clf_type)
            s_time = time()
            simple_clf_feature_model = train_clf(
                simple_clf_model_feature_path, simple_clf_model, train_features, trainY, val_features, valY)
            times['Training SimpleClfFeature_' +
                  options.clf_type] = time() - s_time

        s_time = time()
        simple_clf_feature_train_pred = predict_clf(
            simple_clf_feature_model, train_features)
        times['Train Prediction SimpleClfFeature_' +
              options.clf_type] = time() - s_time
        s_time = time()
        simple_clf_feature_val_pred = predict_clf(
            simple_clf_feature_model, val_features)
        times['Val Prediction SimpleClfFeature_' +
              options.clf_type] = time() - s_time
        s_time = time()
        simple_clf_feature_test_pred = predict_clf(
            simple_clf_feature_model, test_features)
        times['Test Prediction SimpleClfFeature_' +
              options.clf_type] = time() - s_time

    ######## Train Feature Blackbox ########
    if options.include_feature_blackbox:
        # Generators
        trainXf = np.expand_dims(train_features, axis=-1)
        valXf = np.expand_dims(val_features, axis=-1)
        testXf = np.expand_dims(test_features, axis=-1)

        params_feature_simple = {
            'dim': [train_features.shape[1], 1], 'batch_size': 32, 'shuffle': False}
        s_time = time()
        train_feature_generator_simple = DataGenerator_sample(
            np.arange(trainLen), trainXf, trainY, **params_feature_simple)
        times['Train Generator BlackboxFeature'] = time() - s_time
        s_time = time()
        val_feature_generator_simple = DataGenerator_sample(
            np.arange(valLen), valXf, valY, **params_feature_simple)
        times['Val Generator BlackboxFeature'] = time() - s_time
        s_time = time()
        test_feature_generator_simple = DataGenerator_sample(
            np.arange(testLen), testXf, testY, **params_feature_simple)
        times['Test Generator BlackboxFeature'] = time() - s_time

        blackbox_feature_model_path = os.path.join(
            model_path, 'blackbox_classifier_feature.h5' if not options.use_dense else 'blackbox_classifier_feature_dense.h5')
        if os.path.exists(blackbox_feature_model_path) and options.load_feature_blackbox:
            blackbox_feature_model = tf.keras.models.load_model(
                blackbox_feature_model_path)
        else:
            if not options.use_dense:
                blackbox_feature_model = create_model(
                    trainXf.shape[1:], classes)
            else:
                blackbox_feature_model = create_dense_model(
                    trainXf.shape[1:], classes)
            s_time = time()
            blackbox_feature_model = train_blackbox(blackbox_feature_model_path, blackbox_feature_model,
                                                    train_feature_generator_simple, val_feature_generator_simple, epochs=50, verbose=1, workers=1)
            times['Training BlackboxFeature'] = time() - s_time

        s_time = time()
        bfm_train_pred = np.argmax(blackbox_feature_model.predict(
            train_feature_generator_simple), axis=-1)[:trainLen]
        times['Train Prediction BlackboxFeature'] = time() - s_time
        s_time = time()
        bfm_val_pred = np.argmax(blackbox_feature_model.predict(
            val_feature_generator_simple), axis=-1)[:valLen]
        times['Val Prediction BlackboxFeature'] = time() - s_time
        s_time = time()
        bfm_test_pred = np.argmax(blackbox_feature_model.predict(
            test_feature_generator_simple), axis=-1)[:testLen]
        times['Test Prediction BlackboxFeature'] = time() - s_time

    ######## Save Time Benchmark ###########
    if options.store_times:
        print('Save time benchmark')
        times_path = os.path.join(
            model_path, 'timebenchmark_' + setup + '.pickle')
        with open(times_path, 'wb') as f:
            pickle.dump(times, f)

    ######## Accuracy Statistics ###########
    if options.get_statistics or options.save_statistics:
        if options.include_l2:
            get_classification_report(testY, clf_test_pred, verbose=True)
            print('Interpretable')
            int_rep = get_complete_evaluation(trainY, valY, testY,
                                              clf_train_pred, clf_val_pred, clf_test_pred)
            if options.save_statistics:
                write_to_file(os.path.join(
                    setup_path, 'accuracy_report_' + options.clf_type + '.txt'), int_rep)

        if options.include_trivial:
            print('Trivial')
            triv_rep = get_complete_evaluation(
                trainY, valY, testY, trivial_train_preds, trivial_val_preds, trivial_test_preds)
            if options.save_statistics:
                write_to_file(os.path.join(
                    setup_path, 'accuracy_report_trivial.txt'), triv_rep)

        if options.include_blackbox:
            print('Blackbox')
            black_rep = get_complete_evaluation(
                trainY, valY, testY, bm_train_pred, bm_val_pred, bm_test_pred)
            if options.save_statistics:
                write_to_file(os.path.join(
                    model_path, 'accuracy_report.txt'), black_rep)

        if options.include_simple_clf:
            print('SimpleClf')
            simple_rep = get_complete_evaluation(
                trainY, valY, testY, simple_clf_train_pred, simple_clf_val_pred, simple_clf_test_pred)
            if options.save_statistics:
                write_to_file(os.path.join(
                    model_path, 'accuracy_report_simpleClf_' + options.clf_type + '.txt'), simple_rep)

        if options.include_feature_blackbox:
            print('Feature Blackbox')
            black_feature_rep = get_complete_evaluation(
                trainY, valY, testY, bfm_train_pred, bfm_val_pred, bfm_test_pred)
            if options.save_statistics:
                write_to_file(os.path.join(
                    model_path, 'accuracy_report_feature.txt' if not options.use_dense else 'accuracy_report_feature_dense.txt'), black_feature_rep)

        if options.include_feature_simple_clf:
            print('Feature SimpleClf')
            simple_feature_rep = get_complete_evaluation(
                trainY, valY, testY, simple_clf_feature_train_pred, simple_clf_feature_val_pred, simple_clf_feature_test_pred)
            if options.save_statistics:
                write_to_file(os.path.join(model_path, 'accuracy_report_simpleClf_' +
                                           options.clf_type + '_feature.txt'), simple_feature_rep)

        if options.include_l2:
            # currently not in use
            train_mis = get_misclassifications(trainY, clf_train_pred)
            val_mis = get_misclassifications(valY, clf_val_pred)
            test_mis = get_misclassifications(testY, clf_test_pred)

            train_class_means = compute_class_mean(histo_trainX, trainY)
            val_class_means = compute_class_mean(histo_valX, valY)
            test_class_means = compute_class_mean(histo_testX, testY)

    ########### Plot Statistics ############
    if options.include_plots:
        if options.sample_set == 'train':
            sample_set, sample_labels, sample_len, clf_sample_pred, sample_means = trainX, trainY, trainLen. clf_train_pred, train_class_means
        elif options.sample_set == 'val':
            sample_set, sample_labels, sample_len, clf_sample_pred, sample_means = valX, valY, valLen, clf_val_pred, val_class_means
        else:
            sample_set, sample_labels, sample_len, clf_sample_pred, sample_means = testX, testY, testLen, clf_test_pred, test_class_means

        if options.plot_patch:
            # patch + dist
            idx = options.sample_idx
            show_patches = np.fromstring(options.patch_ids, dtype=int, sep=',')

            npc, pps = get_data_patch_stats(sample_len, seqlen, config)
            ids = get_all_patch_params(idx, npc, pps)
            samples = get_all_patch(
                ids, sample_set, sample_len, seqlen, config, options.zero, options.attach, options.notemp)

            patch_dists = patch_model.predict(samples)
            param_list = get_patch_params_list(ids, sample_len, seqlen, config)

            for i in ids[show_patches]:
                plot_patch_and_dist(idx, sample_set[idx], sample_labels[idx], np.argmax(
                    patch_dists[i]), patch_dists[i], param_list[i], patch=i, save=image_path if options.save_plots else None)

        if options.plot_sample:
            # complete sample
            idx = options.sample_idx

            npc, pps = get_data_patch_stats(sample_len, seqlen, config)
            ids = get_all_patch_params(idx, npc, pps)
            samples = get_all_patch(
                ids, sample_set, sample_len, seqlen, config, options.zero, options.attach)

            patch_preds = patch_model.predict(samples)
            plot_series_and_dist(
                idx, sample_set[idx], sample_labels[idx], clf_sample_pred[idx], patch_preds, save=image_path if options.save_plots else None)

        if options.plot_overlay:
            # class overlay
            idx = options.sample_idx
            only_classes = np.fromstring(
                options.overlay_classes, dtype=int, sep=',')
            if len(only_classes) == 0:
                only_classes = None
            ids = get_all_patch_params(idx, npc, pps)
            samples = get_all_patch(ids, sample_set, sample_len, seqlen,
                                    config, options.zero, options.attach, options.notemp)

            patch_dists = patch_model.predict(samples)
            param_list = get_patch_params_list(ids, trainLen, seqlen, config)
            plot_class_overlay(idx, sample_set[idx], sample_labels[idx], clf_sample_pred[idx],
                               patch_dists, param_list, only_classes, save=image_path if options.save_plots else None)

        if options.plot_mean_matrix:
            # class means
            plot_heatmap(
                sample_means, save=image_path if options.save_plots else None)

        if options.plot_mean_diagram:
            if len(sample_mean_classes) == 1:
                plot_class_means(np.expand_dims(
                    sample_means[sample_mean_classes], axis=0), save=image_path if options.save_plots else None)
            else:
                plot_class_means(
                    sample_means[sample_mean_classes], save=image_path if options.save_plots else None)


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    ########### Prepare Data ###############
    parser.add_option("--path", action="store", type=str,
                      dest="path", default='../../data/character_trajectories/dataset_steps-20_timesteps-206.pickle', help="data path")

    ########### Store times ################
    parser.add_option("--store_times", action="store_true",
                      dest="store_times", default=False, help="Store the compute times")

    ############# Patch Data ###############
    parser.add_option("--strides", action="store", type=str,
                      dest="strides", default='10', help="strides for patches")
    parser.add_option("--length", action="store", type=str,
                      dest="length", default='20', help="length for patches")
    parser.add_option("--zero", action="store", type=int,
                      dest="zero", default=False, help="hide information outside patch")
    parser.add_option("--attach", action="store", type=int,
                      dest="attach", default=False, help="attach relevant information channel")
    parser.add_option("--notemp", action="store", type=int,
                      dest="notemp", default=False, help="remove time information")

    ############ Train Level 1 #############
    parser.add_option("--include_l1", action="store_true",
                      dest="include_l1", default=False, help="include level 1 model")
    parser.add_option("--load_l1", action="store_true",
                      dest="load_l1", default=False, help="load the level 1 model")

    ########## Trivial Classifier ##########
    parser.add_option("--include_trivial", action="store_true",
                      dest="include_trivial", default=False, help="include trivial level 2model")
    parser.add_option("--trivial_mode", action="store", type=str,
                      dest="trivial_mode", default='majority', help="majority or occurance based on the problem statement")

    ############ Train Level 2 #############
    parser.add_option("--include_l2", action="store_true",
                      dest="include_l2", default=False, help="include the level 2 model")
    parser.add_option("--load_l2", action="store_true",
                      dest="load_l2", default=False, help="load the level 2 model")
    parser.add_option("--clf_type", action="store", type=str,
                      dest="clf_type", default="svm", help="type for level 2 model (svm, random_forest")

    ############ Train Blackbox ############
    parser.add_option("--use_dense", action="store_true",
                      dest="use_dense", default=False, help="use dense feature blackbox")

    parser.add_option("--include_blackbox", action="store_true",
                      dest="include_blackbox", default=False, help="include the blackbox")
    parser.add_option("--load_blackbox", action="store_true",
                      dest="load_blackbox", default=False, help="load blackbox model")
    parser.add_option("--include_simple_clf", action="store_true",
                      dest="include_simple_clf", default=False, help="include the simple classifier")
    parser.add_option("--load_simple_clf", action="store_true",
                      dest="load_simple_clf", default=False, help="load the simple classifier")

    ############ Feature Settings ##########
    parser.add_option("--compute_features", action="store_true",
                      dest="compute_features", default=False, help="computes the features")
    parser.add_option("--force_compute_features", action="store_true",
                      dest="force_compute_features", default=False, help="forces to compute the features")
    parser.add_option("--feature_mode", action="store", type=str,
                      dest="feature_mode", default='all', help="use 'all' or a 'subset' to compute relevant features")
    parser.add_option("--feature_subset", action="store", type=int,
                      dest="feature_subset", default=500, help="number of smaple sused for subset feature extraction")

    parser.add_option("--include_feature_blackbox", action="store_true",
                      dest="include_feature_blackbox", default=False, help="include the feature blackbox")
    parser.add_option("--load_feature_blackbox", action="store_true",
                      dest="load_feature_blackbox", default=False, help="load feature blackbox model")
    parser.add_option("--include_feature_simple_clf", action="store_true",
                      dest="include_feature_simple_clf", default=False, help="include the feature simple classifier")
    parser.add_option("--load_feature_simple_clf", action="store_true",
                      dest="load_feature_simple_clf", default=False, help="load the feature simple classifier")

    ######## Accuracy Statistics ###########
    parser.add_option("--get_statistics", action="store_true",
                      dest="get_statistics", default=False, help="compute the statistcs")
    parser.add_option("--save_statistics", action="store_true",
                      dest="save_statistics", default=False, help="save the statistcs")

    ########### Plot Statistics ############
    parser.add_option("--include_plots", action="store_true",
                      dest="include_plots", help="include level 1 model")
    parser.add_option("--save_plots", action="store_true",
                      dest="save_plots", default=False, help="Saves the plots")
    parser.add_option("--plot_patch", action="store_true",
                      dest="plot_patch", default=False, help="plot a patch and its classification")
    parser.add_option("--sample_set", action="store", type=str,
                      dest="sample_set", default='test', help="select either the train, val or test set")
    parser.add_option("--patch_ids", action="store", type=str,
                      dest="patch_ids", default=0, help="the indices of the patches")
    parser.add_option("--plot_sample", action="store_true",
                      dest="plot_sample", default=False, help="plot a sample and its classification")
    parser.add_option("--plot_overlay", action="store_true",
                      dest="plot_overlay", default=False, help="plot a sample and the class overlay")
    parser.add_option("--only_classes", action="store", type=str,
                      dest="only_classes", default='', help="select the classes for the overlay")
    parser.add_option("--sample_idx", action="store", type=int,
                      dest="sample_idx", default=0, help="the idx of the sample")
    parser.add_option("--plot_mean_matrix", action="store_true",
                      dest="plot_mean_matrix", default=False, help="plot the mean class matrix")
    parser.add_option("--plot_mean_diagram", action="store_true",
                      dest="plot_mean_diagram", default=False, help="plot the mean class diagram")
    parser.add_option("--mean_classes", action="store", type=str,
                      dest="mean_classes", default='0,1', help="select the classes for the mean diagram")

    # Parse command line options
    (options, args) = parser.parse_args()

    # print options
    print(options)

    process(options)
