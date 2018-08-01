import {
  FETCH_TASK,
  FETCH_TASK_SUCCESS,
  FETCH_TASK_FAILURE,
  FETCH_STATS,
  FETCH_STATS_SUCCESS,
  FETCH_STATS_FAILURE,
  FETCH_ITEMS,
  FETCH_ITEMS_SUCCESS,
  FETCH_ITEMS_FAILURE,
  RECORD_JUDGEMENT,
  RECORD_JUDGEMENT_SUCCESS,
  RECORD_JUDGEMENT_FAILURE,
  SUBMIT_DATA,
  SUBMIT_DATA_FAILURE,
  SUBMIT_DATA_SUCCESS,
  SHOW_NEXT_ITEM,
  LABELLING_COMPLETE,
  SET_BOUNDING_BOX_CLASS,
  CHANGE_SEQUENCE_INPUT,
  CHANGE_DEMO_SCORE_TEXT,
  CHANGE_DEMO_SCORE_URL_SEQUENCE,
  CHANGE_DEMO_SCORE_URL,
  SET_IS_BATCH_VIEW,
} from '../actions';

const task = (state = {
  title: null,
  description: null,
  dataType: null,
  labelType: null,
  classes: null,
  validTokens: [],
  fetching: false,
  errorMsg: null,
  isBatchView: false,
  minTrain: 0,
  minTest: 0,
  minUnsup: 0,
}, action) => {
  switch (action.type) {
    case FETCH_TASK:
      return {
        ...state,
        fetching: true,
      }
    case FETCH_TASK_SUCCESS:
      return {
        ...state,
        title: action.task.title,
        description: action.task.description,
        dataType: action.task.data_type,
        labelType: action.task.label_type,
        classes: action.task.classes,
        validTokens: action.task.valid_tokens,
        fetching: false,
        minTrain: action.task.min_train,
        minTest: action.task.min_test,
        minUnsup: action.task.min_unsup,
      }
    case FETCH_TASK_FAILURE:
      return {
        ...state,
        fetching: false,
        errorMsg: action.errorMsg,
      }
    case SET_IS_BATCH_VIEW:
      return {
        ...state,
        isBatchView: action.isBatchView,
      }
    default:
      return state
  }
}

const judgements = (state = {
  submitting: false,
  sequenceInput: '',
}, action) => {
  switch (action.type) {
    case CHANGE_SEQUENCE_INPUT:
      return {
        ...state,
        sequenceInput: action.sequenceInput,
      }
    case RECORD_JUDGEMENT:
      return {
        ...state,
        submitting: true,
      }
    case RECORD_JUDGEMENT_SUCCESS:
      return {
        ...state,
        errorMsg: null,
        submitting: false,
      }
    case RECORD_JUDGEMENT_FAILURE:
      return {
        ...state,
        submitting: false,
        errorMsg: action.errorMsg,
      }
    default:
      return state
  }
}

const stats = (state = {
  fetching: false,
  history: [],
  labelled: {
    model_labelled: 0,
    test: 0,
    total: 0,
    train: 0,
  },
  unlabelled: 0,
}, action) => {
  switch (action.type) {
    case FETCH_STATS:
      return {
        ...state,
        fetching: true
      }
    case FETCH_STATS_SUCCESS:
      return {
        ...state,
        fetching: false,
        history: action.stats.history,
        labelled: action.stats.labelled,
        unlabelled: action.stats.unlabelled,
        errorMsg: null,
      }
    case FETCH_STATS_FAILURE:
      return {
        ...state,
        fetching: false,
        errorMsg: action.errorMsg,
      }
    default:
      return state
  }
}

const items = (state = {
  fetching: false,
  items: [],
  entropy: [],
  done: false,
  stages: [],
  predictions: [],
  errorMsg: null,
  currentIndex: null,
  currentBoundingBoxClass: null, // Only relevant for bounding box task
}, action) => {
  switch (action.type) {
    case SHOW_NEXT_ITEM:
      return {
        ...state,
        currentIndex: state.currentIndex + 1,
      }
    case SET_BOUNDING_BOX_CLASS:
      return {
        ...state,
        currentBoundingBoxClass: action.boundingBoxClass,
      }
    case FETCH_ITEMS:
      return {
        ...state,
        fetching: true,
      };
    case FETCH_ITEMS_SUCCESS:
      return {
        ...state,
        fetching: false,
        currentIndex: state.currentIndex == null ? 0 : state.currentIndex,
        items: state.items.concat(action.items.batch),
        done: action.items.done,
        entropy: state.entropy.concat(action.items.entropy),
        predictions: action.items.y_prediction ? state.predictions.concat(action.items.y_prediction) : [],
        stages: state.stages.concat(Array(action.items.batch.length).fill(action.items.stage)),
        errorMsg: null,
      };
    case FETCH_ITEMS_FAILURE:
      return {
        ...state,
        fetching: false,
        errorMsg: action.errorMsg,
      };
    case LABELLING_COMPLETE:
      return {
        ...state,
        done: true,
      }
    default:
      return state
  }
}

const demo = (state = {
  submitting: false,
  errorMsg: null,
  scores: null,
  text: '',
  urlSequence: '',
  url: '',
  predictions: [],
}, action) => {
  switch (action.type) {
    case SUBMIT_DATA:
      return {
        ...state,
        submitting: true,
      }
    case SUBMIT_DATA_FAILURE:
      return {
        ...state,
        submitting: false,
        errorMsg: action.errorMsg,
      }
    case SUBMIT_DATA_SUCCESS:
      return {
        ...state,
        predictions: action.response.predictions,
        url: state.urlSequence,
        submitting: false,
      }
    case CHANGE_DEMO_SCORE_TEXT:
      return {
        ...state,
        text: action.text,
      }
    case CHANGE_DEMO_SCORE_URL_SEQUENCE:
      return {
        ...state,
        urlSequence: action.urlSequence,
      }
    default:
      return state
  }
}

export default {
  task, judgements, stats, items, demo,
};
