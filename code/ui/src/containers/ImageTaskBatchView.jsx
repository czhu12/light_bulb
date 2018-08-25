import { connect } from 'react-redux';
import React from 'react';
import { zip } from 'lodash';
import { submitBatchJudgements } from '../actions';

class ImageTaskBatchView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {selected: props.batchItems.items.map(() => true)};
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
    let images = zip(items, selected).map((itemAndSelected, idx) => {
      let item = itemAndSelected[0];
      let src = `/images?image_path=${item['path']}`;
      let selected = itemAndSelected[1];
      if (selected) {
        return (<img onClick={this.onClickImage.bind(this, idx)} className="img-batch-view-selected" src={src} />);
      } else {
        return (<img onClick={this.onClickImage.bind(this, idx)} className="img-batch-view" src={src} />);
      }
    });

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
            )}
            className="btn btn-primary">
            Done
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
  onClickSubmit: (items, selected, targetClass) => {
    let judgements = items.map((item, idx) => {
      return {
        path: item['path'],
        is_target_class: selected[idx],
        target_class: targetClass,
      }
    });

    dispatch(submitBatchJudgements(judgements));
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(ImageTaskBatchView);
