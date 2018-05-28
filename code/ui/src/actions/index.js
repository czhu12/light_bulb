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

export const SET_BOUNDING_BOX_CLASS = 'SET_BOUNDING_BOX_CLASS';
export const setBoundingBoxClass = (boundingBoxClass) => ({
  type: SET_BOUNDING_BOX_CLASS,
  boundingBoxClass,
});

export const CHANGE_SEQUENCE_INPUT = 'CHANGE_SEQUENCE_INPUT';
export const changeSequenceInput = (sequenceInput) => ({
  type: CHANGE_SEQUENCE_INPUT,
  sequenceInput,
});

export function getNextBatch() {
  return (dispatch) => {
    dispatch(fetchItems());
    const query = new URLSearchParams({ prediction: false })

    return fetch(`/batch?${query.toString()}`).then((response) => {
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
    return fetch('/history').then((response) => {
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
    return fetch('/task').then((response) => {
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
    dispatch(submitJudgementToBackend(itemId, judgement, () => {
      if (currentIndex + 4 > items.length) {
        dispatch(getNextBatch())
      }
      dispatch(getStats())
      dispatch(showNextItem())
    }))
  }
}

export function submitJudgementToBackend(itemId, judgement, successCallback) {
  return (dispatch) => {
    dispatch(recordJudgement(itemId, judgement));
    return fetch('/judgements', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: itemId, label: judgement }),
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
