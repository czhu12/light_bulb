import { connect } from 'react-redux';
import React from 'react';
import { zip } from 'lodash';
import ReactDOM from 'react-dom';

import { CLASSIFICATION_COLORS } from '../constants';
import { submitBatchJudgements } from '../actions';
import { createJudgementsFromBatch } from '../utils';

class ImageTaskBatchView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {selected: props.batchItems.items.map(() => true)};
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (prevProps.batchItems.items !== this.props.batchItems.items) {
      // Scroll to top.
    }
  }

  componentWillReceiveProps(nextProps) {
    let selected = nextProps.batchItems.items.map(() => true);
    this.setState({selected: selected});
  }

  onClickImage(idx) {
    let selected = this.state.selected.slice(0);
    selected[idx] = !selected[idx];
    this.setState({selected: selected});
  }

  render() {
    let items = this.props.batchItems.items;
    let selected = this.state.selected;
    let color = CLASSIFICATION_COLORS[this.props.batchItems.targetClass];
    let images = zip(items, selected).map((itemAndSelected, idx) => {
      let item = itemAndSelected[0];
      let src = `/images?image_path=${item['path']}`;
      let selected = itemAndSelected[1];
      if (selected) {
        let style = { borderWidth: '5px', borderStyle: 'solid', borderColor: color }
        return (<img onClick={this.onClickImage.bind(this, idx)} style={style} className="img-batch-view-selected" src={src} />);
      } else {
        return (<img onClick={this.onClickImage.bind(this, idx)} className="img-batch-view" src={src} />);
      }
    });

    let fetchingView = this.props.batchItems.fetching ?
      (<i className="fa fa-spinner fa-spin" aria-hidden="true"></i>) :
        (<i className="fa fa-arrow-right" aria-hidden="true"></i>);

    return (
      <div>
        {images}
        <div className="pull-right">
          <a
            onClick={this.props.onClickSubmit.bind(
              null,
              this.props.batchItems.items,
              this.state.selected,
              this.props.batchItems.targetClass,
              this.props.task.classes,
            )}
            style={{color: '#eee', backgroundColor: color}}
            className="btn btn-lg">
            Submit {fetchingView}
          </a>
        </div>
      </div>
    );
  }
}

const mapStateToProps = state => ({
  task: state.task,
  batchItems: state.batchItems,
});

const mapDispatchToProps = dispatch => ({
  onClickSubmit: (items, selected, targetClass, classes) => {
    const judgements = createJudgementsFromBatch(items, selected, targetClass, classes)
    dispatch(submitBatchJudgements(judgements));
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(ImageTaskBatchView);
