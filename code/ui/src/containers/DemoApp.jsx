import { connect } from 'react-redux';
import React from 'react';
import NavigationBar from './NavigationBar';
import BoundingBoxDemoView from './BoundingBoxDemoView';
import { submitDataToScore, changeDemoScoreUrlSequence, changeDemoScoreText } from '../actions';

class DemoApp extends React.Component {
  render() {
    let demoView = null;
    let firstPrediction = null;
    if (this.props.demo.predictions.length > 0) {
      firstPrediction = this.props.demo.predictions[0];
    }

    if (this.props.task.dataType === 'images') {
      demoView = (
				<div>
          <form>
            <div className="form-group">
              <label>
                {this.props.task.title}
                <b>
                  {firstPrediction !== null ? firstPrediction.toString() : ""}
                </b>
              </label>

              <input
                value={this.props.demo.urlSequence}
                placeholder="Image URL"
                className="form-control"
                rows="3"
                onChange={this.props.onChangeUrlSequence}
              />
            </div>
            <button
              onClick={this.props.submitDataToScore}
              className="btn btn-primary"
            >
              Classify Image
            </button>
          </form>
          <div className="row">
            <div className="col-md-6 offset-md-2">
              <img className="img-fluid" src={this.props.demo.url} />
            </div>
          </div>
				</div>
      )
    } else if (this.props.task.dataType === 'text') {
      demoView = (
        <div>
          <form>
            <div className="form-group">
              <label>{this.props.task.title}</label>
              <textarea
                value={this.props.demo.text}
                className="form-control"
                rows="3"
                onChange={this.props.onChangeText}
              ></textarea>
            </div>
            <button
              onClick={this.props.submitDataToScore}
              className="btn btn-primary">
              Classify Text
            </button>
          </form>
        </div>
      )
    } else if (this.props.task.dataType == 'object_detection') {
      <BoundingBoxDemoView
        url={this.props.demo.url}
        boundingBoxes={firstPrediction}
      />
    }
    return (
      <div>
        <NavigationBar />
        <div className="container">
          { demoView }
        </div>
      </div>
    );
  }
}

const mapStateToProps = state => ({
  task: state.task,
  demo: state.demo,
});


const mapDispatchToProps = dispatch => ({
  submitDataToScore: (e) => {
    e.preventDefault();
    dispatch(submitDataToScore())
  },
  onChangeText: (e) => {
    let text = e.target.value;
    dispatch(changeDemoScoreText(text));
  },
  onChangeUrlSequence: (e) => {
    let urlSequence = e.target.value;
    dispatch(changeDemoScoreUrlSequence(urlSequence));
  },
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(DemoApp);
