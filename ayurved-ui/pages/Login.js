import { useNavigate } from "react-router-dom";
import { useState } from "react";

export default function Login() {
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = () => {
    if (email && password) {
      nav("/dashboard");
    }
  };

  return (
    <div style={{ textAlign: "center", padding: 100 }}>
      <h2>Login</h2>

      <input placeholder="Email" onChange={e => setEmail(e.target.value)} />
      <br /><br />

      <input type="password" placeholder="Password"
        onChange={e => setPassword(e.target.value)} />
      <br /><br />

      <button onClick={handleLogin}>Login</button>

      <p onClick={() => nav("/signup")} style={{ cursor: "pointer" }}>
        Create Account
      </p>
    </div>
  );
}
