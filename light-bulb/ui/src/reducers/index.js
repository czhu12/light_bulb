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
  SET_CURRENT_SELECTED_CLASS,
  CHANGE_SEQUENCE_INPUT,
  CHANGE_DEMO_SCORE_TEXT,
  CHANGE_DEMO_SCORE_URL_SEQUENCE,
  CHANGE_DEMO_SCORE_URL,
  SET_IS_BATCH_VIEW,

  // Separate pathway for dealing with batched data.
  RECORD_JUDGEMENTS,
  RECORD_JUDGEMENTS_SUCCESS,
  RECORD_JUDGEMENTS_FAILURE,
  FETCH_BATCH_ITEMS,
  FETCH_BATCH_ITEMS_SUCCESS,
  FETCH_BATCH_ITEMS_FAILURE,
  BATCH_LABELLING_COMPLETE,
  UPDATE_BATCH_ITEMS_BY_INDEX,
  CHANGE_NAVBAR_SEARCH_QUERY,
} from '../actions';

const task = (state = {
  title: null,
  description: null,
  dataType: null,
  labelType: null,
  classes: null,
  validTokens: [],
  fetching: false,
  template: null,
  errorMsg: null,
  isBatchView: false,
  minTrain: 0,
  minTest: 0,
  minUnsup: 0,
  defaultClass: null,
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
        template: action.task.template,
        description: action.task.description,
        dataType: action.task.data_type,
        labelType: action.task.label_type,
        classes: action.task.classes,
        validTokens: action.task.valid_tokens,
        fetching: false,
        minTrain: action.task.min_train,
        minTest: action.task.min_test,
        minUnsup: action.task.min_unsup,
        defaultClass: action.task.default_class,
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
    case BATCH_LABELLING_COMPLETE:
      // If batch labelling is complete, lets switch back to single item labelling.
      return {
        ...state,
        isBatchView: false,
      };
    default:
      return state
  }
}

const judgements = (state = {
  submitting: false,
  sequenceInput: '',
  timeLastSubmitted: new Date().getTime(),
}, action) => {
  switch (action.type) {
    case CHANGE_SEQUENCE_INPUT:
      return {
        ...state,
        sequenceInput: action.sequenceInput,
      }
    case RECORD_JUDGEMENTS:
      return {
        ...state,
        timeLastSubmitted: new Date().getTime(),
      }
    case RECORD_JUDGEMENT:
      return {
        ...state,
        submitting: true,
        timeLastSubmitted: new Date().getTime(),
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
  averageTimeTaken: 0,
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
        averageTimeTaken: action.stats.average_time_taken,
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

// Batch items currently only supports image classification tasks.
const batchItems = (state = {
  fetching: false,
  items: [],
  targetClass: 0,
  predictions: [],
  done: false,
  errorMsg: null,
  searchQuery: '',
}, action) => {
  switch (action.type) {
    case FETCH_BATCH_ITEMS:
      return {
        ...state,
        fetching: true,
      };
    case UPDATE_BATCH_ITEMS_BY_INDEX:
      let items = state.items.slice();
      items[action.index] = action.item;
      return {
        ...state,
        items,
      };
    case FETCH_BATCH_ITEMS_SUCCESS:
      return {
        ...state,
        fetching: false,
        items: action.items.batch,
        targetClass: action.items.target_class,
        predictions: action.items.predictions,
        done: action.items.done,
        errorMsg: null,
      };
    case FETCH_BATCH_ITEMS_FAILURE:
      return {
        ...state,
        fetching: false,
        errorMsg: action.errorMsg,
      };
    case BATCH_LABELLING_COMPLETE:
      return {
        ...state,
        done: true,
      };
    case CHANGE_NAVBAR_SEARCH_QUERY:
      return {
        ...state,
        searchQuery: action.searchQuery,
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
  currentSelectedClass: null, // Only relevant for bounding box task
}, action) => {
  switch (action.type) {
    case SHOW_NEXT_ITEM:
      return {
        ...state,
        currentIndex: state.currentIndex + 1,
      }
    case SET_CURRENT_SELECTED_CLASS:
      return {
        ...state,
        currentSelectedClass: action.currentSelectedClass,
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
        currentSelectedClass: 0,
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
  task, judgements, stats, items, demo, batchItems,
};
