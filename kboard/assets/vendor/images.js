var accChart;
var fscoreChart;
var lossChart;

function updateData(chart, y, metric) {
  if ((chart.series != undefined) && (y != null)) {
    series = chart.series[0];
    shift = series.data.length > 10000;
    x = series.data.length;
    for (i = 0; i < y.length; i++) {
      series.addPoint([x, y[i]], true, shift);
      x++;
    }
    setTimeout(function() {
      requestData(chart, metric)
    }, 2000);
  }
}

/*
 * Fetches the data from the server. If there is no data in the given chart object,
 * then asks the server for all data so far. If there is data in the chart object,
 * only asks for all items not so far given.
 *
 * Returns a list of values.
 */
function requestData(chart, metric) {
  var x;
  if (chart.series != undefined) {
    x = chart.series[0].data.length;
  } else {
    x = 0;
  }
  $.ajax({
    url: "/data",
    data: {"x": x, "metric": metric},
    success: function(y) {
      updateData(chart, y, metric);
    },
    cache: false
  });
}

function makeChart(metric, title, container, min, max) {
  return new Highcharts.Chart({
    chart: {
      renderTo: container,
      defaultSeriesType: "spline",
      events: {
        load: function() {
          requestData(this, metric);
        }
      }
    },
    title: {
      text: title
    },
    xAxis: {
      tickPixelInterval: 150,
      maxZoom: 20
    },
    yAxis: {
      min: min,
      max: max,
      minPadding: 0.1,
      maxPadding: 0.1,
      title: {
        text: title,
        margin: 80
      }
    },
    series: [{
      name: title,
      data: []
    }]
  });
}

$(document).ready(function() {
  accChart = makeChart("acc", "Accuracy", "accContainer", 0.0, 1.0);
  fscoreChart = makeChart("fscore", "FScore", "fscoreContainer", 0.0, 1.0);
  lossChart = makeChart("loss", "Loss", "lossContainer", 0.0, undefined);
});
