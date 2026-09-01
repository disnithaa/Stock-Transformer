export default function MonthlyTable({ monthly }) {
  return (
    <div className="chart-wrap" style={{ overflowX: "auto" }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Month</th>
            <th>MAE</th>
            <th>RMSE</th>
            <th>R²</th>
            <th>Directional acc.</th>
          </tr>
        </thead>
        <tbody>
          {monthly.map((row) => (
            <tr key={row.month}>
              <td>{row.month}</td>
              <td>{row.mae.toFixed(2)}</td>
              <td>{row.rmse.toFixed(2)}</td>
              <td>{row.r2 == null ? "—" : row.r2.toFixed(3)}</td>
              <td>{row.da.toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
