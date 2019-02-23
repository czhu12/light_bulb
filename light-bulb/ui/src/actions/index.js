import 'url-search-params-polyfill';

export const FETCH_TASK = 'TASK/FETCH';
export const fetchTask = () => ({
  type: FETCH_TASK,
});

export const FETCH_TASK_SUCCESS = 'TASK/FETCH/SUCCESS';
export const fetchTaskSuccess = response => ({
  type: FETCH_TASK_SUCCESS,
  task: response,
});

export const FETCH_TASK_FAILURE = 'TASK/FETCH/FAILURE';
export const fetchTaskFailure = error => ({
  type: FETCH_TASK_FAILURE,
  errorMsg: error,
});

export const FETCH_STATS = 'STATS/FETCH';
export const fetchStats = () => ({
  type: FETCH_STATS,
});

export const FETCH_STATS_SUCCESS = 'STATS/FETCH/SUCCESS';
export const fetchStatsSuccess = response => ({
  type: FETCH_STATS_SUCCESS,
  stats: response,
});

export const FETCH_STATS_FAILURE = 'STATS/FETCH/FAILURE';
export const fetchStatsFailure = error => ({
  type: FETCH_STATS_FAILURE,
  errorMsg: error,
});

export const FETCH_ITEMS = 'ITEMS/FETCH';
export const fetchItems = () => ({
  type: FETCH_ITEMS,
});

export const FETCH_ITEMS_SUCCESS = 'ITEMS/FETCH/SUCCESS';
export const fetchItemsSuccess = response => ({
  type: FETCH_ITEMS_SUCCESS,
  items: response,
});

export const FETCH_ITEMS_FAILURE = 'ITEMS/FETCH/FAILURE';
export const fetchItemsFailure = error => ({
  type: FETCH_ITEMS_FAILURE,
  errorMsg: error,
});

export const RECORD_JUDGEMENT = 'JUDGEMENT/RECORD';
export const recordJudgement = (itemId, judgement) => ({
  type: RECORD_JUDGEMENT,
  itemId,
  judgement,
});

export const RECORD_JUDGEMENT_SUCCESS = 'JUDGEMENT/RECORD/SUCCESS';
export const recordJudgementSuccess = itemId => ({
  type: RECORD_JUDGEMENT_SUCCESS,
  itemId,
});

export const RECORD_JUDGEMENT_FAILURE = 'JUDGEMENT/RECORD/FAILURE';
export const recordJudgementFailure = (itemId, error) => ({
  type: RECORD_JUDGEMENT_FAILURE,
  itemId,
  errorMsg: error,
});

export const SHOW_NEXT_ITEM = 'SHOW_NEXT_ITEM';
export const showNextItem = () => ({
  type: SHOW_NEXT_ITEM
});

export const LABELLING_COMPLETE = 'LABELLING_COMPLETE';
export const labellingComplete = () => ({
  type: LABELLING_COMPLETE
});

export const SET_CURRENT_SELECTED_CLASS = 'SET_CURRENT_SELECTED_CLASS';
export const setCurrentSelectedClass = (currentSelectedClass) => ({
  type: SET_CURRENT_SELECTED_CLASS,
  currentSelectedClass,
});

export const CHANGE_SEQUENCE_INPUT = 'CHANGE_SEQUENCE_INPUT';
export const changeSequenceInput = (sequenceInput) => ({
  type: CHANGE_SEQUENCE_INPUT,
  sequenceInput,
});

// Params: force_stage, sample_size, prediction
export function getNextBatch(params) {
  return (dispatch) => {
    dispatch(fetchItems());
    const query = new URLSearchParams(params)

    return fetch(`/api/batch?${query.toString()}`).then((response) => {
      if (!response.ok) {
        throw Error(response.statusText);
      }
      return response.json();
    }).then((json) => {
      dispatch(fetchItemsSuccess(json))
    }).catch(error => dispatch(fetchItemsFailure(`Error fetching batch items: ${error.message}`)));
  }
}

export function getStats() {
  return (dispatch) => {
    dispatch(fetchStats());
    return fetch('/api/history').then((response) => {
      if (!response.ok) {
        throw Error(response.statusText);
      }
      return response.json();
    }).then((json) => {
      dispatch(fetchStatsSuccess(json))
    }).catch(error => dispatch(fetchStatsFailure(`Error fetching stats: ${error.message}`)));
  }
}

export function getTask() {
  return (dispatch) => {
    dispatch(fetchTask());
    return fetch('/api/task').then((response) => {
      if (!response.ok) {
        throw Error(response.statusText);
      }
      return response.json();
    }).then((json) => {
      dispatch(fetchTaskSuccess(json))
    }).catch(error => dispatch(fetchTaskFailure(`Error fetching stats: ${error.message}`)));
  }
}

export function submitJudgement(judgement) {
  return (dispatch, getState) => {
    const state = getState();
    if (state.judgements.submitting) { // Double clicked
      console.log('Double called submitJudgement.');
      return;
    }

    const currentIndex = state.items.currentIndex;
    const items = state.items.items;
    const itemId = items[currentIndex]['path'];

    let timeTaken = new Date().getTime() - state.judgements.timeLastSubmitted;
    dispatch(submitJudgementToBackend(itemId, judgement, timeTaken, () => {
      // If 4 away from last item.
      if (currentIndex + 4 > items.length) {
        dispatch(getNextBatch())
      }
      dispatch(getStats())
      dispatch(showNextItem())
    }))
  }
}

export function submitJudgementToBackend(itemId, judgement, timeTaken, successCallback) {
  return (dispatch) => {
    dispatch(recordJudgement(itemId, judgement));
    return fetch('/api/judgements', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: itemId, label: judgement, time_taken: timeTaken }),
    })
      .then((response) => {
        if (!response.ok) {
          throw Error(response.statusText);
        }
        return response.json();
      })
      .then((json) => {
        if ('error' in json) {
          throw new Error(json['error'])
        } else {
          dispatch(recordJudgementSuccess(itemId));
          successCallback();
        }
      })
      .catch(error => dispatch(recordJudgementFailure(itemId, error.message)));
  };
}

export const SUBMIT_DATA = 'DATA/SUBMIT';
export const submitData = (data) => ({
  type: SUBMIT_DATA,
  data,
});

export const SUBMIT_DATA_SUCCESS = 'DATA/SUBMIT/SUCCESS';
export const submitDataSuccess = (response) => ({
  type: SUBMIT_DATA_SUCCESS,
  response,
});

export const SUBMIT_DATA_FAILURE = 'DATA/SUBMIT/FAILURE';
export const submitDataFailure = (error) => ({
  type: SUBMIT_DATA_FAILURE,
  errorMsg: error,
});

export const CHANGE_DEMO_SCORE_URL = 'CHANGE_DEMO_SCORE_URL';
export const changeDemoScoreUrl = (url) => ({
  type: CHANGE_DEMO_SCORE_URL,
  url,
});

export const CHANGE_DEMO_SCORE_URL_SEQUENCE = 'CHANGE_DEMO_SCORE_URL_SEQUENCE';
export const changeDemoScoreUrlSequence = (urlSequence) => ({
  type: CHANGE_DEMO_SCORE_URL_SEQUENCE,
  urlSequence,
});

export const CHANGE_DEMO_SCORE_TEXT = 'CHANGE_DEMO_SCORE_TEXT';
export const changeDemoScoreText = (text) => ({
  type: CHANGE_DEMO_SCORE_TEXT,
  text,
});

export function submitDataToScore() {
  return (dispatch, getState) => {
    let state = getState();
    let body = null;
    if (state.task.dataType === 'images') {
      body = { type: 'images', urls: [state.demo.urlSequence] }
    } else if (state.task.dataType === 'text') {
      body = { type: 'text', texts: [state.demo.text] }
    } else if (state.task.dataType === 'json') {
      // Here, we have to tokenize the text. We should tokenize with punctuation
      let tokenized = state.demo.text.split(/\W+/);
      body = { type: 'text', texts: [tokenized] }
    }

    dispatch(submitData(body));

    return fetch('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then((response) => {
        if (!response.ok) {
          throw Error(response.statusText);
        }
        return response.json();
      })
      .then((json) => {
        if ('error' in json) {
          throw new Error(json['error'])
        } else {
          dispatch(submitDataSuccess(json));
        }
      })
      .catch(error => dispatch(submitDataFailure(error.message)));
  }
}

export const RECORD_JUDGEMENTS = 'JUDGEMENTS/RECORD';
export const recordJudgements = (judgements) => ({
  type: RECORD_JUDGEMENTS,
  judgements,
});

export const RECORD_JUDGEMENTS_SUCCESS = 'JUDGEMENTS/RECORD/SUCCESS';
export const recordJudgementsSuccess = judgements => ({
  type: RECORD_JUDGEMENTS_SUCCESS,
  judgements,
});

export const RECORD_JUDGEMENTS_FAILURE = 'JUDGEMENTS/RECORD/FAILURE';
export const recordJudgementsFailure = (judgements, error) => ({
  type: RECORD_JUDGEMENTS_FAILURE,
  judgements,
  errorMsg: error,
});

export const FETCH_BATCH_ITEMS = 'ITEMS/BATCH/FETCH';
export const fetchBatchItems = () => ({
  type: FETCH_BATCH_ITEMS,
});

export const FETCH_BATCH_ITEMS_SUCCESS = 'ITEMS/BATCH/FETCH/SUCCESS';
export const fetchBatchItemsSuccess = response => ({
  type: FETCH_BATCH_ITEMS_SUCCESS,
  items: response,
});

export const FETCH_BATCH_ITEMS_FAILURE = 'ITEMS/BATCH/FETCH/FAILURE';
export const fetchBatchItemsFailure = error => ({
  type: FETCH_BATCH_ITEMS_FAILURE,
  errorMsg: error,
});

export const BATCH_LABELLING_COMPLETE = 'BATCH_LABELLING_COMPLETE';
export const batchLabellingComplete = () => ({
  type: BATCH_LABELLING_COMPLETE
});

export const UPDATE_BATCH_ITEMS_BY_INDEX = 'UPDATE_BATCH_ITEMS_BY_INDEX';
export const updateBatchItemByIndex = (index, item) => ({
  type: UPDATE_BATCH_ITEMS_BY_INDEX,
  item: item,
  index: index,
});

export function submitBatchJudgements(judgements, isSearch) {
  return (dispatch, getState) => {
    const state = getState();
    if (state.judgements.submitting) { // Double clicked
      console.log('Double called submitJudgements.');
      return;
    }

    let timeTaken = (new Date().getTime() - state.judgements.timeLastSubmitted) / judgements.length;
    dispatch(submitBatchJudgementsToBackend(judgements, timeTaken, isSearch));
  }
}

export function submitBatchJudgementsToBackend(judgements, timeTaken, isSearch) {
  return (dispatch) => {
    dispatch(recordJudgements(judgements));
    return fetch('/api/judgements/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ labels: judgements, time_taken: timeTaken }),
    })
      .then((response) => {
        if (!response.ok) {
          throw Error(response.statusText);
        }
        return response.json();
      })
      .then((json) => {
        if ('error' in json) {
          throw new Error(json['error'])
        } else {
          dispatch(recordJudgementsSuccess(judgements));
          if (isSearch) {
            dispatch(submitSearchBatchQuery());
          } else {
            dispatch(fetchNextBatchItemsBatch());
          }
          dispatch(getStats());
        }
      })
      .catch(error => dispatch(recordJudgementsFailure(judgements, error.message)));
  };
}

// Batch labelling for images and text
export const SET_IS_BATCH_VIEW = 'SET_IS_BATCH_VIEW';
export const fetchNextBatchItemsBatch = () => {
  return (dispatch) => {
    dispatch(fetchBatchItems());

    return fetch(`/api/batch_items_batch`).then((response) => {
      if (!response.ok) {
        throw Error(response.statusText);
      }
      return response.json();
    }).then((json) => {
      if (json['done']) {
        dispatch(batchLabellingComplete(json));
      } else {
        dispatch(fetchBatchItemsSuccess(json));
      }
    }).catch(error => dispatch(fetchBatchItemsFailure(`Error fetching batch items: ${error.message}`)));
  }
};

export const setIsBatchView = (isBatchView) => {
  return {
    type: SET_IS_BATCH_VIEW,
    isBatchView,
  }
};

export const CHANGE_NAVBAR_SEARCH_QUERY = 'CHANGE_NAVBAR_SEARCH_QUERY';
export const changeNavbarSearchQuery = (searchQuery) => {
  return {
    type: CHANGE_NAVBAR_SEARCH_QUERY,
    searchQuery,
  }
};

export const submitSearchBatchQuery = () => {
  return (dispatch, getState) => {
    let state = getState();
    let searchQuery = state.batchItems.searchQuery;
    dispatch(fetchBatchItems());
    return fetch(`/api/batch_items_batch?search_query=${searchQuery}`).then((response) => {
      if (!response.ok) {
        throw Error(response.statusText);
      }
      return response.json();
    }).then((json) => {
      dispatch(setIsBatchView(true));
      dispatch(fetchBatchItemsSuccess(json));
    }).catch(error => dispatch(fetchBatchItemsFailure(`Error fetching batch items: ${error.message}`)));
  }
};

export const FETCH_DATASET = 'TASK/DATASET';
export const fetchDataset = () => ({
  type: FETCH_DATASET,
});

export const FETCH_DATASET_SUCCESS = 'DATASET/FETCH/SUCCESS';
export const fetchDatasetSuccess = ({dataset}) => ({
  type: FETCH_DATASET_SUCCESS,
  dataset: dataset,
});

export const FETCH_DATASET_FAILURE = 'DATASET/FETCH/FAILURE';
export const fetchDatasetFailure = error => ({
  type: FETCH_DATASET_FAILURE,
  errorMsg: error,
});

export function getNextDatasetPage() {
  return (dispatch, getState) => {
    const state = getState();
    if (state.dataset.submitting) {
      return;
    }

    dispatch(fetchDataset());
    let body = { page: state.dataset.page, page_size: 20 };
    if (state.labelled != null) {
      body['labelled'] = state.labelled;
    }
    const query = new URLSearchParams(body);

    return fetch(`/api/dataset?${query.toString()}`).then((response) => {
      if (!response.ok) {
        throw Error(response.statusText);
      }
      return response.json();
    }).then((json) => {
      dispatch(fetchDatasetSuccess(json));
    }).catch(error => dispatch(fetchDatasetFailure(`Error fetching dataset: ${error.message}`)));
  }
}

