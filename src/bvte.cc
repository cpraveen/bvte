/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Andrea Bonito, Sebastian Pauletti.
 */

// The initial code for this is taken from step-38 in deal.II 8.3.0

// @sect3{Include files}

// If you've read through step-4 and step-7, you will recognize that we have
// used all of the following include files there already. Consequently, we
// will not explain their meaning here again.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

const double a_rk[] = {0.0, 3.0/4.0, 1.0/3.0};
const double b_rk[] = {1.0, 1.0/4.0, 2.0/3.0};

namespace VTE
{
   using namespace dealii;
   enum SolverType { DIRECT, CG };
   
   // Initial vorticity distribution
   // x = R*cos(lambda), y = R*sin(lambda), z = R*sin(theta)
   // -pi/2 <= theta <= pi/2,   -pi <= lambda <= pi
   template <int dim>
   class InitialCondition  : public Function<dim>
   {
   public:
      InitialCondition ();
      
      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
      
   private:
      unsigned int n;
      double a, omg, K;
   };
   
   template <int dim>
   InitialCondition<dim>::InitialCondition ()
   :
   Function<dim>()
   {
      n = 4;
      a = n*n + 3*n + 2;
      omg = 7.8480e-6;
      K = 7.8480e-6;
   }

   template <>
   double
   InitialCondition<3>::value (const Point<3> &p,
                               const unsigned int) const
   {
      const double R = p.norm();
      const double lambda = atan2 (p[1], p[0]);
      const double theta = asin (p[2]/R);
      return 2.0 * omg * sin(theta)
             - K * a * sin(theta) * std::pow(cos(theta),n) * cos(n*lambda);
   }

   // @sect3{The <code>VTEProblem</code> class template}

   // This class is almost exactly similar to the <code>LaplaceProblem</code>
   // class in step-4.

   // The essential differences are these:
   //
   // - The template parameter now denotes the dimensionality of the embedding
   //   space, which is no longer the same as the dimensionality of the domain
   //   and the triangulation on which we compute. We indicate this by calling
   //   the parameter @p spacedim , and introducing a constant @p dim equal to
   //   the dimensionality of the domain -- here equal to
   //   <code>spacedim-1</code>.
   // - All member variables that have geometric aspects now need to know about
   //   both their own dimensionality as well as that of the embedding
   //   space. Consequently, we need to specify both of their template
   //   parameters one for the dimension of the mesh @p dim, and the other for
   //   the dimension of the embedding space, @p spacedim. This is exactly what
   //   we did in step-34, take a look there for a deeper explanation.
   // - We need an object that describes which kind of mapping to use from the
   //   reference cell to the cells that the triangulation is composed of. The
   //   classes derived from the Mapping base class do exactly this. Throughout
   //   most of deal.II, if you don't do anything at all, the library assumes
   //   that you want an object of kind MappingQ1 that uses a (bi-, tri-)linear
   //   mapping. In many cases, this is quite sufficient, which is why the use
   //   of these objects is mostly optional: for example, if you have a
   //   polygonal two-dimensional domain in two-dimensional space, a bilinear
   //   mapping of the reference cell to the cells of the triangulation yields
   //   an exact representation of the domain. If you have a curved domain, one
   //   may want to use a higher order mapping for those cells that lie at the
   //   boundary of the domain -- this is what we did in step-11, for
   //   example. However, here we have a curved domain, not just a curved
   //   boundary, and while we can approximate it with bilinearly mapped cells,
   //   it is really only prudent to use a higher order mapping for all
   //   cells. Consequently, this class has a member variable of type MappingQ;
   //   we will choose the polynomial degree of the mapping equal to the
   //   polynomial degree of the finite element used in the computations to
   //   ensure optimal approximation, though this iso-parametricity is not
   //   required.
   template <int spacedim>
   class VTEProblem
   {
   public:
      VTEProblem (const unsigned degree = 2);
      void run ();

   private:
      static const int dim = spacedim-1;

      void make_grid_and_dofs ();
      void initialize_vorticity ();
      void assemble_mass_matrix();
      void assemble_streamfun_matrix ();
      void assemble_streamfun_rhs ();
      void assemble_vte_rhs ();
      void update_vte (const unsigned int rk);
      void solve_streamfun ();
      void output_results (const double elapsed_time) const;
      void compute_error () const;


      double       cfl, dt;
      unsigned int n_rk_stages;
      SolverType   solver_type;
      
      Triangulation<dim,spacedim>   triangulation;

      FE_Q<dim,spacedim>            fe_stream;
      DoFHandler<dim,spacedim>      dof_handler_stream;

      FE_DGQArbitraryNodes<dim,spacedim> fe_vort;
      DoFHandler<dim,spacedim>      dof_handler_vort;

      MappingQ<dim,spacedim>        mapping;

      SparsityPattern               sparsity_pattern;
      SparseMatrix<double>          system_matrix;
      PreconditionSSOR<>            preconditioner;


      Vector<double>                streamfun;
      Vector<double>                system_rhs;

      // Vorticity related variables
      Vector<double>                vorticity;
      Vector<double>                vorticity_old;
      Vector<double>                vorticity_rhs;

      std::vector< Vector<double> > inv_mass_matrix;
   };


   // @sect3{Equation data}

   // Next, let us define the classes that describe the exact solution and the
   // right hand sides of the problem. This is in analogy to step-4 and step-7
   // where we also defined such objects. Given the discussion in the
   // introduction, the actual formulas should be self-explanatory. A point of
   // interest may be how we define the value and gradient functions for the 2d
   // and 3d cases separately, using explicit specializations of the general
   // template. An alternative to doing it this way might have been to define
   // the general template and have a <code>switch</code> statement (or a
   // sequence of <code>if</code>s) for each possible value of the spatial
   // dimension.
   template <int dim>
   class Solution  : public Function<dim>
   {
   public:
      Solution () : Function<dim>() {}

      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;

      virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                      const unsigned int  component = 0) const;

   };


   template <>
   double
   Solution<3>::value (const Point<3> &p,
                       const unsigned int) const
   {
      return (std::sin(numbers::PI * p(0)) *
              std::cos(numbers::PI * p(1))*exp(p(2)));
   }


   template <>
   Tensor<1,3>
   Solution<3>::gradient (const Point<3>   &p,
                          const unsigned int) const
   {
      using numbers::PI;

      Tensor<1,3> return_value;

      return_value[0] = PI *cos(PI * p(0))*cos(PI * p(1))*exp(p(2));
      return_value[1] = -PI *sin(PI * p(0))*sin(PI * p(1))*exp(p(2));
      return_value[2] = sin(PI * p(0))*cos(PI * p(1))*exp(p(2));

      return return_value;
   }

   // @sect3{Implementation of the <code>VTEProblem</code> class}

   // The rest of the program is actually quite unspectacular if you know
   // step-4. Our first step is to define the constructor, setting the
   // polynomial degree of the finite element and mapping, and associating the
   // DoF handler to the triangulation:
   template <int spacedim>
   VTEProblem<spacedim>::VTEProblem (const unsigned degree)
      :
      fe_stream (QGaussLobatto<1>(degree+1)),
      dof_handler_stream (triangulation),
      fe_vort (QGauss<1>(degree+1)),
      dof_handler_vort (triangulation),
      mapping (degree)
   {
      n_rk_stages = 3;
      cfl = 0.5;
      solver_type = DIRECT;
      
      cfl = cfl/(2.0*degree+1);
   }


   // @sect4{VTEProblem::make_grid_and_dofs}

   // The next step is to create the mesh, distribute degrees of freedom, and
   // set up the various variables that describe the linear system. All of
   // these steps are standard with the exception of how to create a mesh that
   // describes a surface. We could generate a mesh for the domain we are
   // interested in, generate a triangulation using a mesh generator, and read
   // it in using the GridIn class. Or, as we do here, we generate the mesh
   // using the facilities in the GridGenerator namespace.
   //
   // In particular, what we're going to do is this (enclosed between the set
   // of braces below): we generate a <code>spacedim</code> dimensional mesh
   // for the half disk (in 2d) or half ball (in 3d), using the
   // GridGenerator::half_hyper_ball function. This function sets the boundary
   // indicators of all faces on the outside of the boundary to zero for the
   // ones located on the perimeter of the disk/ball, and one on the straight
   // part that splits the full disk/ball into two halves. The next step is the
   // main point: The GridGenerator::extract_boundary_mesh function creates a mesh
   // that consists of those cells that are the faces of the previous mesh,
   // i.e. it describes the <i>surface</i> cells of the original (volume)
   // mesh. However, we do not want all faces: only those on the perimeter of
   // the disk or ball which carry boundary indicator zero; we can select these
   // cells using a set of boundary indicators that we pass to
   // GridGenerator::extract_boundary_mesh.
   //
   // There is one point that needs to be mentioned. In order to refine a
   // surface mesh appropriately if the manifold is curved (similarly to
   // refining the faces of cells that are adjacent to a curved boundary), the
   // triangulation has to have an object attached to it that describes where
   // new vertices should be located. If you don't attach such a boundary
   // object, they will be located halfway between existing vertices; this is
   // appropriate if you have a domain with straight boundaries (e.g. a
   // polygon) but not when, as here, the manifold has curvature. So for things
   // to work properly, we need to attach a manifold object to our (surface)
   // triangulation, in much the same way as we've already done in 1d for the
   // boundary. We create such an object (with indefinite, <code>static</code>,
   // lifetime) at the top of the function and attach it to the triangulation
   // for all cells with boundary indicator zero that will be created
   // henceforth.
   //
   // The final step in creating the mesh is to refine it a number of
   // times. The rest of the function is the same as in previous tutorial
   // programs.
   template <int spacedim>
   void VTEProblem<spacedim>::make_grid_and_dofs ()
   {

      {
         Triangulation<spacedim> volume_mesh;
         GridGenerator::hyper_ball (volume_mesh);

         std::set<types::boundary_id> boundary_ids;
         boundary_ids.insert (0);

         GridGenerator::extract_boundary_mesh (volume_mesh,
                                               triangulation,
                                               boundary_ids);
      }
      
      static SphericalManifold<dim,spacedim> surface_description;
      triangulation.set_all_manifold_ids(0);
      triangulation.set_manifold (0, surface_description);

      triangulation.refine_global(4);

      // Save mesh to file for visualization
      GridOut grid_out;
      std::ofstream grid_file("grid.vtk");
      grid_out.write_vtk(triangulation, grid_file);
      std::cout << "Grid has been saved into grid.vtk" << std::endl;

      std::cout << "Surface mesh has " << triangulation.n_active_cells()
                << " cells."
                << std::endl;

      dof_handler_stream.distribute_dofs (fe_stream);

      std::cout << "Surface mesh has " << dof_handler_stream.n_dofs()
                << " degrees of freedom."
                << std::endl;

      DynamicSparsityPattern dsp (dof_handler_stream.n_dofs(), dof_handler_stream.n_dofs());
      DoFTools::make_sparsity_pattern (dof_handler_stream, dsp);
      sparsity_pattern.copy_from (dsp);

      system_matrix.reinit (sparsity_pattern);

      streamfun.reinit (dof_handler_stream.n_dofs());
      system_rhs.reinit (dof_handler_stream.n_dofs());

      dof_handler_vort.distribute_dofs (fe_vort);
      vorticity.reinit (dof_handler_vort.n_dofs());
      vorticity_old.reinit (dof_handler_vort.n_dofs());
      vorticity_rhs.reinit (dof_handler_vort.n_dofs());

      inv_mass_matrix.resize(triangulation.n_cells());
      for (unsigned int c=0; c<triangulation.n_cells(); ++c)
         inv_mass_matrix[c].reinit(fe_vort.dofs_per_cell);
      
      // set cell number
      unsigned int index=0;
      for (typename Triangulation<dim,spacedim>::active_cell_iterator
           cell=triangulation.begin_active();
           cell!=triangulation.end(); ++cell, ++index)
         cell->set_user_index (index);
   }

   //------------------------------------------------------------------------------
   // Set initial condition for vorticity
   //------------------------------------------------------------------------------
   template <int spacedim>
   void VTEProblem<spacedim>::initialize_vorticity ()
   {
      VectorTools::interpolate (mapping,
                                dof_handler_vort,
                                InitialCondition<spacedim>(),
                                vorticity);
   }
   
   //------------------------------------------------------------------------------
   // Assemble mass matrix for each cell
   // Invert it and store. Mass matrix is diagonal, so we store only the diagonal
   // entries.
   //------------------------------------------------------------------------------
   template <int spacedim>
   void VTEProblem<spacedim>::assemble_mass_matrix ()
   {
      std::cout << "Constructing mass matrix ...\n";

      QGauss<dim>  quadrature_formula(fe_vort.degree+1);

      FEValues<dim,spacedim> fe_values (mapping, fe_vort, quadrature_formula,
                                        update_values | update_JxW_values);

      const unsigned int   dofs_per_cell = fe_vort.dofs_per_cell;
      const unsigned int   n_q_points    = quadrature_formula.size();

      Vector<double>   cell_matrix (dofs_per_cell);

      for (typename DoFHandler<dim,spacedim>::active_cell_iterator
           cell = dof_handler_vort.begin_active(),
           endc = dof_handler_vort.end();
           cell!=endc; ++cell)
      {
         fe_values.reinit (cell);
         cell_matrix = 0.0;

         for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
               cell_matrix(i) += fe_values.shape_value (i, q_point) *
                                 fe_values.shape_value (i, q_point) *
                                 fe_values.JxW (q_point);

         // Invert cell_matrix
         unsigned int c = cell->user_index();
         for (unsigned int i=0; i<dofs_per_cell; ++i)
            inv_mass_matrix[c](i) = 1.0/cell_matrix(i);
      }

   }

   // @sect4{VTEProblem::assemble_system}

   // The following is the central function of this program, assembling the
   // matrix that corresponds to the surface Laplacian (Laplace-Beltrami
   // operator). Maybe surprisingly, it actually looks exactly the same as for
   // the regular Laplace operator discussed in, for example, step-4. The key
   // is that the FEValues::shape_gradient function does the magic: It returns
   // the surface gradient $\nabla_K \phi_i(x_q)$ of the $i$th shape function
   // at the $q$th quadrature point. The rest then does not need any changes
   // either:
   template <int spacedim>
   void VTEProblem<spacedim>::assemble_streamfun_matrix ()
   {
      std::cout << "Construcing streamfunction matrix ...\n";

      system_matrix = 0;

      const QGauss<dim>  quadrature_formula(2*fe_stream.degree);
      FEValues<dim,spacedim> fe_values (mapping, fe_stream, quadrature_formula,
                                        update_gradients           |
                                        update_JxW_values);

      const unsigned int        dofs_per_cell = fe_stream.dofs_per_cell;
      const unsigned int        n_q_points    = quadrature_formula.size();

      FullMatrix<double>        cell_matrix (dofs_per_cell, dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      for (typename DoFHandler<dim,spacedim>::active_cell_iterator
           cell = dof_handler_stream.begin_active(),
           endc = dof_handler_stream.end();
           cell!=endc; ++cell)
      {
         cell_matrix = 0;

         fe_values.reinit (cell);

         for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
               for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                  cell_matrix(i,j) += fe_values.shape_grad(i,q_point) *
                                      fe_values.shape_grad(j,q_point) *
                                      fe_values.JxW(q_point);

         cell->get_dof_indices (local_dof_indices);
         for (unsigned int i=0; i<dofs_per_cell; ++i)
         {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
               system_matrix.add (local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i,j));
         }
      }

      // Build preconditioner
      preconditioner.initialize(system_matrix, 1.2);
   }

   //------------------------------------------------------------------------------
   // Assemble rhs of streamfunction problem
   //------------------------------------------------------------------------------
   template <int spacedim>
   void VTEProblem<spacedim>::assemble_streamfun_rhs ()
   {
      system_rhs = 0;

      const QGauss<dim>  quadrature_formula(2*fe_stream.degree);
      FEValues<dim,spacedim> fe_values (mapping, fe_stream, quadrature_formula,
                                        update_values              |
                                        update_quadrature_points   |
                                        update_JxW_values);
      FEValues<dim,spacedim> fe_values_vort (mapping,
                                             fe_vort,
                                             quadrature_formula,
                                             update_values);

      const unsigned int        dofs_per_cell = fe_stream.dofs_per_cell;
      const unsigned int        n_q_points    = quadrature_formula.size();

      Vector<double>            cell_rhs (dofs_per_cell);

      std::vector<double>       vorticity_values(n_q_points);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      for (typename DoFHandler<dim,spacedim>::active_cell_iterator
           cell = dof_handler_stream.begin_active(),
           endc = dof_handler_stream.end();
           cell!=endc; ++cell)
      {
         cell_rhs = 0;
         fe_values.reinit (cell);

         typename DoFHandler<dim,spacedim>::active_cell_iterator
         cell_vort (&triangulation,
                    cell->level(),
                    cell->index(),
                    &dof_handler_vort);
         fe_values_vort.reinit (cell_vort);
         fe_values_vort.get_function_values (vorticity, vorticity_values);

         for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
               cell_rhs(i) -= vorticity_values[q_point] *
                              fe_values.shape_value(i,q_point) *
                              fe_values.JxW(q_point);

         cell->get_dof_indices (local_dof_indices);
         for (unsigned int i=0; i<dofs_per_cell; ++i)
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }

   }

   //------------------------------------------------------------------------------
   // Solve for the stream function
   //------------------------------------------------------------------------------
   template <int spacedim>
   void VTEProblem<spacedim>::solve_streamfun ()
   {
      static bool first_time = true;
      
      switch (solver_type)
      {
         case DIRECT:
         {
            static SparseDirectUMFPACK  solver;
            if(first_time)
            {
               solver.initialize (system_matrix);
               first_time = false;
            }
            solver.vmult (streamfun, system_rhs);
            break;
         }
            
         case CG:
         {
            SolverControl solver_control (streamfun.size(),
                                          1e-8 * system_rhs.l2_norm());
            SolverCG<>    cg (solver_control);
            
            // make initial guess zero since we repeatedly solve this
            streamfun = 0;
            
            cg.solve (system_matrix,
                      streamfun,
                      system_rhs,
                      preconditioner);
            break;
         }
            
         default:
            AssertThrow(false, ExcMessage("Unknown solver type !!!"));
      }
   }


   //------------------------------------------------------------------------------
   // Assemble right hand side of VTE
   //------------------------------------------------------------------------------
   template <int spacedim>
   void VTEProblem<spacedim>::assemble_vte_rhs ()
   {
      vorticity_rhs = 0;
      
      const QGauss<dim>  quadrature_formula(fe_vort.degree + 1);
      FEValues<dim,spacedim> fe_values_vort (mapping, fe_vort, quadrature_formula,
                                             update_values            |
                                             update_gradients         |
                                             update_quadrature_points |
                                             update_JxW_values);
      const unsigned int dofs_per_cell = fe_vort.dofs_per_cell;
      const unsigned int n_q_points    = quadrature_formula.size();
      
      Vector<double>            cell_rhs (dofs_per_cell);
      Vector<double>            cell_rhs_nbr (dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices_nbr (dofs_per_cell);

      std::vector<double> vorticity_values (n_q_points);
      
      FEValues<dim,spacedim> fe_values_stream (mapping, fe_stream, quadrature_formula,
                                               update_gradients);
      std::vector< Tensor<1,spacedim> > stream_gradient_values (n_q_points, Tensor<1,spacedim>());
      
      // Face related
      const QGauss<dim-1> quadrature_face (fe_vort.degree + 1);
      FEFaceValues<dim,spacedim> fe_face_values_vort (mapping,
                                                      fe_vort,
                                                      quadrature_face,
                                                      update_values |
                                                      update_quadrature_points |
                                                      update_normal_vectors |
                                                      update_JxW_values);
      FEFaceValues<dim,spacedim> fe_face_values_vort_nbr (mapping,
                                                          fe_vort,
                                                          quadrature_face,
                                                          update_values);
      FEFaceValues<dim,spacedim> fe_face_values_stream (mapping,
                                                        fe_stream,
                                                        quadrature_face,
                                                        update_gradients);
      FEFaceValues<dim,spacedim> fe_face_values_stream_nbr (mapping,
                                                            fe_stream,
                                                            quadrature_face,
                                                            update_gradients);
      
      const unsigned int n_face_q_points    = quadrature_face.size();
      std::vector< Tensor<1,spacedim> > stream_gradient_values_face     (n_face_q_points, Tensor<1,spacedim>());
      std::vector< Tensor<1,spacedim> > stream_gradient_values_face_nbr (n_face_q_points, Tensor<1,spacedim>());
      std::vector<double> vorticity_values_face     (n_face_q_points);
      std::vector<double> vorticity_values_face_nbr (n_face_q_points);

      double dt_local = 1.0e20;
      
      for (typename DoFHandler<dim,spacedim>::active_cell_iterator
           cell = dof_handler_vort.begin_active(),
           endc = dof_handler_vort.end();
           cell!=endc; ++cell)
      {
         const unsigned int cell_no = cell->user_index();
         double velmax = 0;

         cell_rhs = 0;
         
         fe_values_vort.reinit (cell);
         fe_values_vort.get_function_values (vorticity, vorticity_values);
         
         typename DoFHandler<dim,spacedim>::active_cell_iterator
         cell_stream (&triangulation,
                      cell->level(),
                      cell->index(),
                      &dof_handler_stream);
         fe_values_stream.reinit (cell_stream);
         fe_values_stream.get_function_gradients (streamfun, stream_gradient_values);
         
         for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         {
            const Point<spacedim> &radial_unit_vector = fe_values_vort.quadrature_point (q_point);
            Tensor<1,spacedim> vel = cross_product_3d (radial_unit_vector, stream_gradient_values[q_point]);
            velmax = std::max(velmax, vel.norm());
            for (unsigned int i=0; i<dofs_per_cell; ++i)
               cell_rhs(i) -= vorticity_values[q_point] *
                              vel * fe_values_vort.shape_grad (i, q_point) *
                              fe_values_vort.JxW(q_point);
         }
         
         const double h = cell->diameter();
         dt_local = std::min (dt_local, h / (velmax + 1.0e-14));
         
         for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if(cell->neighbor(f)->has_children() == false)
            {
               const unsigned int cell_nbr_no = cell->neighbor(f)->user_index();
               
               cell_rhs_nbr = 0;
               
               if(cell->neighbor(f)->level() == cell->level() &&
                  cell_no < cell_nbr_no)
               {
                  fe_face_values_vort.reinit (cell, f);
                  fe_face_values_vort_nbr.reinit(cell->neighbor(f),
                                                 cell->neighbor_of_neighbor(f));

                  fe_face_values_vort.get_function_values (vorticity, vorticity_values_face);
                  fe_face_values_vort_nbr.get_function_values (vorticity, vorticity_values_face_nbr);
                  
                  fe_face_values_stream.reinit (cell_stream, f);
                  fe_face_values_stream_nbr.reinit(cell_stream->neighbor(f),
                                                   cell_stream->neighbor_of_neighbor(f));
                  
                  fe_face_values_stream.get_function_gradients (streamfun, stream_gradient_values_face);
                  fe_face_values_stream_nbr.get_function_gradients (streamfun, stream_gradient_values_face_nbr);
                  
                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                  {
                     const Point<spacedim> &radial_unit_vector = fe_face_values_vort.quadrature_point (q_point);
                     Tensor<1,spacedim> vel = cross_product_3d (radial_unit_vector, stream_gradient_values_face[q_point]);
                     Tensor<1,spacedim> vel_nbr = cross_product_3d (radial_unit_vector, stream_gradient_values_face_nbr[q_point]);
                     double veln = 0.5 * (vel + vel_nbr) * fe_face_values_vort.normal_vector (q_point);
                     double flux = (veln > 0) ? veln * vorticity_values_face[q_point]
                                              : veln * vorticity_values_face_nbr[q_point];

                     for (unsigned int i=0; i<dofs_per_cell; ++i)
                     {
                        cell_rhs(i) += flux *
                                       fe_face_values_vort.shape_value (i, q_point) *
                                       fe_face_values_vort.JxW(q_point);
                        cell_rhs_nbr(i) -= flux *
                                           fe_face_values_vort_nbr.shape_value (i, q_point) *
                                           fe_face_values_vort.JxW(q_point);
                     }
                  }
                  
                  // Add local contribute to rhs vector
                  cell->neighbor(f)->get_dof_indices (local_dof_indices_nbr);
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                     vorticity_rhs(local_dof_indices_nbr[i]) += cell_rhs_nbr (i) * inv_mass_matrix[cell_nbr_no](i);
               }
               else if(cell->neighbor_is_coarser(f))
               {
                  AssertThrow (false, ExcNotImplemented());
               }
            }
         
         // Add local contribute to rhs vector
         cell->get_dof_indices (local_dof_indices);
         for (unsigned int i=0; i<dofs_per_cell; ++i)
            vorticity_rhs(local_dof_indices[i]) += cell_rhs(i) * inv_mass_matrix[cell_no](i);
      }
      
      if(dt <= 0.0)
      {
         dt = cfl * dt_local;
      }
   }

   // Perform rk'th stage of SSPRK scheme
   template <int spacedim>
   void VTEProblem<spacedim>::update_vte (const unsigned int rk)
   {
      for(unsigned int i=0; i<dof_handler_vort.n_dofs(); ++i)
         vorticity(i) = a_rk[rk] * vorticity_old(i) +
                        b_rk[rk] * (vorticity(i) - dt * vorticity_rhs(i));
   }
   
   // @sect4{VTEProblem::output_result}

   // This is the function that generates graphical output from the
   // solution. Most of it is boilerplate code, but there are two points worth
   // pointing out:
   //
   // - The DataOut::add_data_vector function can take two kinds of vectors:
   //   Either vectors that have one value per degree of freedom defined by the
   //   DoFHandler object previously attached via DataOut::attach_dof_handler;
   //   and vectors that have one value for each cell of the triangulation, for
   //   example to output estimated errors for each cell. Typically, the
   //   DataOut class knows to tell these two kinds of vectors apart: there are
   //   almost always more degrees of freedom than cells, so we can
   //   differentiate by the two kinds looking at the length of a vector. We
   //   could do the same here, but only because we got lucky: we use a half
   //   sphere. If we had used the whole sphere as domain and $Q_1$ elements,
   //   we would have the same number of cells as vertices and consequently the
   //   two kinds of vectors would have the same number of elements. To avoid
   //   the resulting confusion, we have to tell the DataOut::add_data_vector
   //   function which kind of vector we have: DoF data. This is what the third
   //   argument to the function does.
   // - The DataOut::build_patches function can generate output that subdivides
   //   each cell so that visualization programs can resolve curved manifolds
   //   or higher polynomial degree shape functions better. We here subdivide
   //   each element in each coordinate direction as many times as the
   //   polynomial degree of the finite element in use.
   template <int spacedim>
   void VTEProblem<spacedim>::output_results (const double elapsed_time) const
   {
      static unsigned int count = 0;
      
      DataOut<dim,DoFHandler<dim,spacedim> > data_out;
//      data_out.attach_dof_handler (dof_handler_stream);
//      data_out.add_data_vector (streamfun,
//                                "streamfun",
//                                DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
      data_out.attach_dof_handler (dof_handler_vort);
      data_out.add_data_vector (vorticity,
                                "vorticity",
                                DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
      data_out.build_patches (mapping,
                              mapping.get_degree());

      std::string filename = "sol-" + Utilities::int_to_string(count, 4) + ".vtk";
      std::ofstream output (filename.c_str());
      DataOutBase::VtkFlags flags(elapsed_time, count);
      data_out.set_flags(flags);
      data_out.write_vtk (output);
      ++count;
   }



   // @sect4{VTEProblem::compute_error}

   // This is the last piece of functionality: we want to compute the error in
   // the numerical solution. It is a verbatim copy of the code previously
   // shown and discussed in step-7. As mentioned in the introduction, the
   // <code>Solution</code> class provides the (tangential) gradient of the
   // solution. To avoid evaluating the error only a superconvergence points,
   // we choose a quadrature rule of sufficiently high order.
   template <int spacedim>
   void VTEProblem<spacedim>::compute_error () const
   {
      Vector<float> difference_per_cell (triangulation.n_active_cells());
      VectorTools::integrate_difference (mapping, dof_handler_stream, streamfun,
                                         Solution<spacedim>(),
                                         difference_per_cell,
                                         QGauss<dim>(2*fe_stream.degree+1),
                                         VectorTools::H1_norm);

      std::cout << "H1 error = "
                << difference_per_cell.l2_norm()
                << std::endl;
   }



   // @sect4{VTEProblem::run}

   // The last function provides the top-level logic. Its contents are
   // self-explanatory:
   template <int spacedim>
   void VTEProblem<spacedim>::run ()
   {
      make_grid_and_dofs();
      assemble_mass_matrix ();
      assemble_streamfun_matrix ();
      initialize_vorticity ();
      output_results (0);
      
      unsigned int iter = 0;
      double t  = 0;
      double Tf = 1.0e6;
      while (t < Tf)
      {
         vorticity_old = vorticity;
         
         dt = -1.0; // dt is computed inside assemble_vte_rhs
         
         for(unsigned int rk=0; rk<n_rk_stages; ++rk)
         {
            assemble_streamfun_rhs ();
            solve_streamfun ();
            assemble_vte_rhs ();
            update_vte (rk);
         }
         
         t += dt; ++iter;
         std::cout << "Iter = " << iter
                   << ",  t = " << t
                   << ",  dt = " << dt << std::endl;
         
         if(iter%10 == 0) output_results (t);
      }
   }
}


// @sect3{The main() function}

// The remainder of the program is taken up by the <code>main()</code>
// function. It follows exactly the general layout first introduced in step-6
// and used in all following tutorial programs:
int main ()
{
   try
   {
      using namespace dealii;
      using namespace VTE;

      deallog.depth_console (0);

      unsigned int degree = 2;
      VTEProblem<3> vte (degree);
      vte.run();
   }
   catch (std::exception &exc)
   {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
   }
   catch (...)
   {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
   }

   return 0;
}
