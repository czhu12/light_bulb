import React from 'react';
import ReactDOM from 'react-dom';

import thunk from 'redux-thunk';
import { createStore, applyMiddleware, compose, combineReducers } from 'redux';
import { Provider } from 'react-redux';

import createHistory from 'history/createBrowserHistory';
import { Route } from 'react-router';

import { ConnectedRouter, routerReducer, routerMiddleware } from 'react-router-redux';

import LabelAppContainer from './containers/LabelApp';
import DemoAppContainer from './containers/DemoApp';
import DatasetContainer from './containers/DatasetApp';
import { getTask, getNextBatch, getStats } from './actions';
import reducer from './reducers';

const history = createHistory();
const middleware = [
  thunk,
  routerMiddleware(history),
];

const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
const store = createStore(
  combineReducers({
    ...reducer,
    router: routerReducer,
  }),
  composeEnhancers(applyMiddleware(...middleware)),
);
const rootEl = document.getElementById('root');
// eslint-disable-next-line react/no-render-return-value
const render = () => ReactDOM.render(
  // eslint-disable-next-line react/jsx-filename-extension
  <Provider store={store}>
    <ConnectedRouter history={history}>
      <div>
        <Route exact path="/" component={LabelAppContainer} />
        <Route exact path="/demo" component={DemoAppContainer} />
        <Route exact path="/dataset" component={DatasetContainer} />
      </div>
    </ConnectedRouter>
  </Provider>,
  rootEl,
);

render();
store.subscribe(render);
store.dispatch(getTask());
// TODO: Fix this hack
if (window.location.href.indexOf('dataset') !== -1) {
  store.dispatch(getNextBatch({sample_size: 100, force_stage: 'TRAIN', prediction: 'true'}));
} else {
  store.dispatch(getNextBatch());
}
store.dispatch(getStats());
