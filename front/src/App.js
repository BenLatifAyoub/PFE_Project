// App.jsx
import React from 'react'
import { Routes, Route } from 'react-router-dom'
import WelcomePage from './Pages/WelcomePage'
import LoginPage from './Pages/LoginPage'
import SignUpPage from './Pages/SignUpPage'
import HomePage from './Pages/HomePage'
import AnalyzePage from './Pages/AnalyzePage'
import ChatPage from './Pages/ChatPage'
import CoursePage from './Pages/courseQuiz'
import UserProfilePage from './Pages/update'
import GeneratePage from './Pages/generate'

function App() {
  return (
    <Routes>
      <Route path="/" element={<WelcomePage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignUpPage />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/analyze" element={<AnalyzePage />} />
      <Route path="/chat" element={<ChatPage />} />
      <Route path="/course" element={<CoursePage/>} />
      <Route path="/update" element={<UserProfilePage/>} />
      <Route path="/generate" element={<GeneratePage />} />
      {/* Add more routes as needed */}
    </Routes>
  )
}

export default App
