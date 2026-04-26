import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const nav = useNavigate();

  const modules = [
    {
      name: "Ayurveda Recommendation",
      desc: "Get formulations based on symptoms",
      route: "/recommend"
    },
    {
      name: "Dosha Detector",
      desc: "Find your body type",
      route: "/dosha"
    },
    {
      name: "Diet Planner",
      desc: "Personalized diet based on dosha",
      route: "/diet"
    }
  ];

  return (
    <div style={{ fontFamily: "serif" }}>
      
      {/* Header */}
      <div style={{
        background: "#7A3A0A",
        color: "#FFF",
        padding: "40px",
        textAlign: "center"
      }}>
        <h1>🌿 Ayurveda Wellness Dashboard 🌿</h1>
        <p>Your personalized Ayurvedic assistant</p>
      </div>

      {/* Cards */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
        gap: 20,
        padding: 40
      }}>
        {modules.map((m, i) => (
          <div
            key={i}
            onClick={() => nav(m.route)}
            style={{
              border: "1px solid #E5D3B3",
              borderRadius: 16,
              padding: 20,
              cursor: "pointer",
              background: "#FFF8F0",
              transition: "0.3s"
            }}
          >
            <h2>{m.name}</h2>
            <p>{m.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
