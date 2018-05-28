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
  SHOW_NEXT_ITEM,
  LABELLING_COMPLETE,
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
      }
    case FETCH_TASK_FAILURE:
      return {
        ...state,
        fetching: false,
        errorMsg: action.errorMsg,
      }
    default:
      return state
  }
}

const judgements = (state = {
  submitting: false,
}, action) => {
  switch (action.type) {
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
        history: action.history,
        labelled: action.labelled,
        unlabelled: action.unlabelled,
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
  entropy: 0,
  done: false,
  stage: 'TRAIN',
  yPrediction: [],
  errorMsg: null,
  currentIndex: null,
}, action) => {
  switch (action.type) {
    case SHOW_NEXT_ITEM:
      return {
        ...state,
        currentIndex: state.currentIndex + 1,
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
        entropy: action.items.entropy,
        yPrediction: state.yPrediction.concat(action.items.y_prediction),
        stage: action.items.stage,
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

export default {
  task, judgements, stats, items,
};
