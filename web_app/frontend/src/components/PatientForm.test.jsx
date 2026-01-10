import { render, screen } from '@testing-library/react';
import PatientForm from './PatientForm';
import { describe, it, expect } from 'vitest';
import React from 'react';

describe('PatientForm', () => {
  it('renders input fields correctly', () => {
    // Mock onUpdate prop
    const mockUpdate = () => {};
    
    render(<PatientForm onUpdate={mockUpdate} />);
    
    expect(screen.getByPlaceholderText('Ex: John Doe')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Ex: P-12345')).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });
});
