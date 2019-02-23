/*eslint-env jquery*/
import { chunk, zip } from 'lodash';
import { connect } from 'react-redux';
import { shortenText } from '../utils';
import { getNextDatasetPage } from '../actions';
import NavigationBar from './NavigationBar';
import React from 'react';

class DatasetApp extends React.Component {
  componentDidMount() {
    window.addEventListener('scroll', this.handleScroll.bind(this));
  }

  componentWillUnmount() {
    window.removeEventListener('scroll', this.handleScroll.bind(this));
  }

  handleScroll(e) {
    if ($(window).scrollTop() + $(window).height() == $(document).height()) {
      this.props.onScrollToBottom();
    }
  }

  _cardView(item) {
    let path = item['path'];
    let textView = null;
    if (this.props.task.dataType === 'text') {
      textView = <div>{shortenText(item['text'], 25)}</div>
    }

    let imgView = null;
    if (this.props.task.dataType === 'images') {
      let imgPath = `/images?image_path=${path}`;
      imgView = (<img className="card-img-top" src={ imgPath } alt={ path } />);
    }
    let labelView = null;
    console.log(item);
    if (item['labelled']) {
      labelView = (<span>Label:<span className="badge badge-secondary">{item['label']}</span><br/></span>);
    }

    let file = path.split('/')[path.split('/').length - 1];
		return (
			<div className="card card-dataset" style={{width: "85%"}}>
        {imgView}
				<div className="card-body">
          {textView}
          <br />
          <div className="help-block card-text">
            {labelView}
            File: {file}
          </div>
				</div>
			</div>
		);
  }

  render() {
    let items = this.props.dataset.dataset;
    let itemRows = chunk(items, 2).map((rowItems) => {
      return rowItems.map((item) => {
        return (
					<div className="col-md-6">
						{this._cardView(item)}
					</div>
				);
      });
    }).map((row) => {
      return <div className="row">{row}</div>;
    });
    return (
      <div>
        <NavigationBar />
        {itemRows}
      </div>
    );
  }
}

const mapStateToProps = state => ({
  dataset: state.dataset,
  task: state.task,
});


const mapDispatchToProps = dispatch => ({
  onScrollToBottom: () => {
    dispatch(getNextDatasetPage());
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(DatasetApp);
