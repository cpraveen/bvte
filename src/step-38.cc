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

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <fstream>
#include <iostream>

const double a_rk[] = {0.0, 3.0/4.0, 1.0/3.0};
const double b_rk[] = {1.0, 1.0/4.0, 2.0/3.0};

namespace Step38
{
   using namespace dealii;

   //------------------------------------------------------------------------------
   // Class for integrating rhs using MeshWorker
   //------------------------------------------------------------------------------
   template <int spacedim>
   class RHSIntegrator
   {
   private:
      static const unsigned int dim = spacedim-1;

   public:
      RHSIntegrator (const DoFHandler<dim,spacedim> &dof_handler)
         : dof_info (dof_handler) {};

      MeshWorker::IntegrationInfoBox<dim,spacedim> info_box;
      MeshWorker::DoFInfo<dim,spacedim> dof_info;
      MeshWorker::Assembler::ResidualSimple< Vector<double> > assembler;
   };

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
      static const unsigned int dim = spacedim-1;

      void make_grid_and_dofs ();
      void assemble_mass_matrix();
      void assemble_streamfun_matrix ();
      void assemble_streamfun_rhs ();
      void solve ();
      void setup_mesh_worker (RHSIntegrator<spacedim> &);
      void output_results () const;
      void compute_error () const;


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

      typedef MeshWorker::DoFInfo<dim,spacedim> DoFInfo;
      typedef MeshWorker::IntegrationInfo<dim,spacedim> CellInfo;

      static void integrate_cell_term (DoFInfo &dinfo, CellInfo &info);
      static void integrate_boundary_term (DoFInfo &dinfo, CellInfo &info);
      static void integrate_face_term (DoFInfo &dinfo1, DoFInfo &dinfo2,
                                       CellInfo &info1, CellInfo &info2);
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



   template <int dim>
   class RightHandSide : public Function<dim>
   {
   public:
      RightHandSide () : Function<dim>() {}

      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
   };


   template <>
   double
   RightHandSide<3>::value (const Point<3> &p,
                            const unsigned int /*component*/) const
   {
      using numbers::PI;

      Tensor<2,3> hessian;

      hessian[0][0] = -PI*PI*sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
      hessian[1][1] = -PI*PI*sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
      hessian[2][2] = sin(PI*p(0))*cos(PI*p(1))*exp(p(2));

      hessian[0][1] = -PI*PI*cos(PI*p(0))*sin(PI*p(1))*exp(p(2));
      hessian[1][0] = -PI*PI*cos(PI*p(0))*sin(PI*p(1))*exp(p(2));

      hessian[0][2] = PI*cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
      hessian[2][0] = PI*cos(PI*p(0))*cos(PI*p(1))*exp(p(2));

      hessian[1][2] = -PI*sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
      hessian[2][1] = -PI*sin(PI*p(0))*sin(PI*p(1))*exp(p(2));

      Tensor<1,3> gradient;
      gradient[0] = PI * cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
      gradient[1] = - PI * sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
      gradient[2] = sin(PI*p(0))*cos(PI*p(1))*exp(p(2));

      Point<3> normal = p;
      normal /= p.norm();

      return (- trace(hessian)
              + 2 * (gradient * normal)
              + (hessian * normal) * normal);
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
      fe_vort (QGaussLobatto<1>(degree+1)),
      dof_handler_vort (triangulation),
      mapping (degree)
   {}


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
      static SphericalManifold<dim,spacedim> surface_description;

      {
         Triangulation<spacedim> volume_mesh;
         GridGenerator::hyper_ball (volume_mesh);

         std::set<types::boundary_id> boundary_ids;
         boundary_ids.insert (0);

         GridGenerator::extract_boundary_mesh (volume_mesh,
                                               triangulation,
                                               boundary_ids);
      }
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
   }

   //------------------------------------------------------------------------------
   // Assemble mass matrix for each cell
   // Invert it and store
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

      unsigned int c = 0;
      for (typename DoFHandler<dim,spacedim>::active_cell_iterator
           cell = dof_handler_vort.begin_active(),
           endc = dof_handler_vort.end();
           cell!=endc; ++cell, ++c)
      {
         fe_values.reinit (cell);
         cell_matrix = 0.0;

         for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
               cell_matrix(i) += fe_values.shape_value (i, q_point) *
                                 fe_values.shape_value (i, q_point) *
                                 fe_values.JxW (q_point);

         // Invert cell_matrix
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

   // Assemble rhs of streamfunction problem
   template <int spacedim>
   void VTEProblem<spacedim>::assemble_streamfun_rhs ()
   {
      system_rhs = 0;

      const QGauss<dim>  quadrature_formula(2*fe_stream.degree);
      FEValues<dim,spacedim> fe_values (mapping, fe_stream, quadrature_formula,
                                        update_values              |
                                        update_quadrature_points   |
                                        update_JxW_values);

      const unsigned int        dofs_per_cell = fe_stream.dofs_per_cell;
      const unsigned int        n_q_points    = quadrature_formula.size();

      Vector<double>            cell_rhs (dofs_per_cell);

      std::vector<double>       rhs_values(n_q_points);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      const RightHandSide<spacedim> rhs;

      for (typename DoFHandler<dim,spacedim>::active_cell_iterator
           cell = dof_handler_stream.begin_active(),
           endc = dof_handler_stream.end();
           cell!=endc; ++cell)
      {
         cell_rhs = 0;

         fe_values.reinit (cell);

         rhs.value_list (fe_values.get_quadrature_points(), rhs_values);

         for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
               cell_rhs(i) += fe_values.shape_value(i,q_point) *
                              rhs_values[q_point]*
                              fe_values.JxW(q_point);

         cell->get_dof_indices (local_dof_indices);
         for (unsigned int i=0; i<dofs_per_cell; ++i)
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }

   }

   // @sect4{VTEProblem::solve}

   // The next function is the one that solves the linear system. Here, too, no
   // changes are necessary:
   template <int spacedim>
   void VTEProblem<spacedim>::solve ()
   {
      SolverControl solver_control (streamfun.size(),
                                    1e-7 * system_rhs.l2_norm());
      SolverCG<>    cg (solver_control);

      cg.solve (system_matrix,
                streamfun,
                system_rhs,
                preconditioner);
   }

   //------------------------------------------------------------------------------
   // Create mesh worker for integration
   //------------------------------------------------------------------------------
   template <int spacedim>
   void VTEProblem<spacedim>::setup_mesh_worker (RHSIntegrator<spacedim> &rhs_integrator)
   {
      std::cout << "Setting up mesh worker ...\n";

      MeshWorker::IntegrationInfoBox<dim,spacedim> &info_box = rhs_integrator.info_box;
      MeshWorker::DoFInfo<dim,spacedim> &dof_info = rhs_integrator.dof_info;
      MeshWorker::Assembler::ResidualSimple< Vector<double> > &
      assembler = rhs_integrator.assembler;

      const unsigned int n_gauss_points = fe_vort.degree+1;
      info_box.initialize_gauss_quadrature(n_gauss_points,
                                           n_gauss_points,
                                           n_gauss_points);

      // Add solution vector to info_box
//      AnyData solution_data;
//      solution_data.add< Vector<double>* > (&vorticity, "vorticity");
//      info_box.cell_selector.add     ("vorticity", true, false, false);
//      info_box.boundary_selector.add ("vorticity", true, false, false);
//      info_box.face_selector.add     ("vorticity", true, false, false);

      info_box.initialize_update_flags ();
      info_box.add_update_flags_all      (update_quadrature_points);
      info_box.add_update_flags_cell     (update_gradients);
      info_box.add_update_flags_boundary (update_values);
      info_box.add_update_flags_face     (update_values);

      Vector<double> dummy;
      //info_box.initialize (fe_vort, mapping, solution_data, dummy);
      info_box.initialize (fe_vort, mapping);

      // Attach rhs vector to assembler
      AnyData rhs;
      rhs.add< Vector<double>* > (&vorticity_rhs, "vorticity_rhs");
      assembler.initialize (rhs);
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
   void VTEProblem<spacedim>::output_results () const
   {
      DataOut<dim,DoFHandler<dim,spacedim> > data_out;
      data_out.attach_dof_handler (dof_handler_stream);
      data_out.add_data_vector (streamfun,
                                "streamfun",
                                DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
      data_out.build_patches (mapping,
                              mapping.get_degree());

      std::string filename ("solution-");
      filename += static_cast<char>('0'+spacedim);
      filename += "d.vtk";
      std::ofstream output (filename.c_str());
      data_out.write_vtk (output);
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

      RHSIntegrator<spacedim> rhs_integrator (dof_handler_vort);
      setup_mesh_worker (rhs_integrator);

      double t  = 0;
      double Tf = 1.0;
      while (t < Tf)
      {
         assemble_streamfun_rhs ();
         solve ();
         output_results ();
         compute_error ();
         return;
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
      using namespace Step38;

      deallog.depth_console (0);

      VTEProblem<3> vte;
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
